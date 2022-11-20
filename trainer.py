
import os
import wandb
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from dotenv import load_dotenv
from utils.metrics import compute_metrics
from utils.scheduler import LinearWarmupScheduler

class Trainer :

    def __init__(
        self, 
        args,
        model,
        device,
        train_dataloader, 
        eval_dataloader=None
    ) :
        self.args = args
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

    def train(self, ) :
        
        args = self.args

        train_data_iterator = iter(self.train_dataloader)
        total_steps = len(self.train_dataloader) * args.epochs if args.max_steps == -1 else args.max_steps
        warmup_steps = int(total_steps * args.warmup_ratio)

        loss_fn = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = LinearWarmupScheduler(optimizer, total_steps, warmup_steps)
        
        load_dotenv(dotenv_path="wandb.env")
        WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
        wandb.login(key=WANDB_AUTH_KEY)

        name = f"EP:{args.epochs}_BS:{args.train_batch_size}_LR:{args.learning_rate}_WR:{args.warmup_ratio}_WD:{args.weight_decay}"
        wandb.init(
            entity="sangha0411",
            project="bert4rec",
            group=f"ai-ground",
            name=name
        )

        training_args = {
            "epochs": args.epochs, 
            "total_steps" : total_steps,
            "warmup_steps" : warmup_steps,
            "batch_size": args.train_batch_size, 
            "learning_rate": args.learning_rate, 
            "weight_decay": args.weight_decay, 
        }
        wandb.config.update(training_args)

        self.model.to(self.device)
        vocab_size = self.model.config.vocab_size

        for step in tqdm(range(total_steps)) :

            try :
                data = next(train_data_iterator)
            except StopIteration :
                train_data_iterator = iter(self.train_dataloader)
                data = next(train_data_iterator)

            optimizer.zero_grad()

            age_input, gender_input = data['age'], data['gender']
            age_input = age_input.long().to(self.device)
            gender_input = gender_input.long().to(self.device)

            album_input, genre_input, country_input = data['album_input'], data['genre_input'], data['country_input']
            album_input = album_input.long().to(self.device)
            genre_input = genre_input.long().to(self.device)
            country_input = country_input.long().to(self.device)

            logits = self.model(
                album_input=album_input, 
                genre_input=genre_input,
                country_input=country_input,
                age_input=age_input,
                gender_input=gender_input,
            )

            labels = data['labels'].long().to(self.device)
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1,))

            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % args.logging_steps == 0 and step > 0 :
                current_lr = scheduler.get_last_lr()[0]
                log = {'train/step' : step, 'train/loss' : loss.item(), 'train/lr' : current_lr}
                wandb.log(log)
                print(log)

            if args.do_eval :
                if step % args.eval_steps == 0 and step > 0 :
                    print('\nValidation at %d step' %step)
                    self.evaluate()
            else :
                if step % args.save_steps == 0 and step > 0 :
                    model_path = os.path.join(args.save_dir, f'checkpoint-{step}.pt')        
                    torch.save(self.model.state_dict(), model_path)

        if args.do_eval :
            self.evaluate()
        else :
            model_path = os.path.join(args.save_dir, f'checkpoint-{total_steps}.pt')        
            torch.save(self.model.state_dict(), model_path)
            
        wandb.finish()

    def evaluate(self) :
        self.model.eval()

        with torch.no_grad() :
            
            eval_predictions, eval_labels = [], []
            for eval_data in tqdm(self.eval_dataloader) :

                age_input, gender_input = eval_data['age'], eval_data['gender']
                age_input = age_input.long().to(self.device)
                gender_input = gender_input.long().to(self.device)

                album_input, genre_input, country_input = eval_data['album_input'], eval_data['genre_input'], eval_data['country_input']
                album_input = album_input.long().to(self.device)
                genre_input = genre_input.long().to(self.device)
                country_input = country_input.long().to(self.device)

                logits = self.model(
                    album_input=album_input, 
                    genre_input=genre_input,
                    country_input=country_input,
                    age_input=age_input,
                    gender_input=gender_input,
                )

                logits = logits[:,-1,:].detach().cpu().numpy()
                logits = np.argsort(-logits, axis=-1)
                
                eval_predictions.extend(logits.tolist())
                eval_labels.extend(eval_data['labels'])

            eval_log = compute_metrics(eval_predictions, eval_labels)
            eval_log = {'eval/' + k : v for k, v in eval_log.items()}
            wandb.log(eval_log)
            print(eval_log)

        self.model.train()