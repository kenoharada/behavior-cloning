import argparse
import os
import sys

import numpy as np
import torch
import wandb
from src.models import Policy
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset import BCDataset
from r3m import load_r3m_reproduce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=64)
    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--wandb_entity", type=str, default='kenoharada')
    parser.add_argument("--pickle_data_path", type=str, default='/home/keno.harada/behavior-cloning/experiments/data.pkl')
    args = parser.parse_args()
    
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    train_dataset = BCDataset(pickle_file=args.pickle_data_path)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Policy(obs_dim=args.embed_dim, act_dim=args.action_dim, hidden_sizes=(args.hidden_dim, args.hidden_dim)).to(device)
    
    rep = load_r3m_reproduce("r3m").to('cuda')
    rep.eval()
    optimizer = Adam(params=model.parameters(), lr=args.lr, amsgrad=True)
    best_loss = sys.maxsize

    exp_dir_name = os.path.abspath(__file__).split('/')[-2]
    project_name = f'{exp_dir_name}'

    wandb.init(project=project_name, config=args, entity=args.wandb_entity)
    wandb.run.name = f'lr{args.lr}_hidden{args.hidden_dim}_{wandb.run.id}'
    wandb.run.save()
    wandb.config.update(args)

    model.train()
    loss_fn = torch.nn.MSELoss()
    for epoch in range(1, args.epochs + 1):
        wandb_log_dict = dict()
        for obs, action in train_dataloader:
            obs = obs.to('cuda')
            action = action.to('cuda')
            optimizer.zero_grad()
            with torch.no_grad():
                out = rep(obs)
            out =  model(out)
            loss = loss_fn(out, action)
        
            loss.backward()
            optimizer.step()
        _loss = loss.item()
        wandb_log_dict['loss'] = _loss
        wandb.log(wandb_log_dict)
        if _loss < best_loss:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best_model.pt"))
            # https://docs.wandb.ai/guides/track/advanced/save-restore#saving-files
            wandb.save(os.path.join(wandb.run.dir, "best_model.pt"), base_path=wandb.run.dir)
            best_loss = _loss
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir,f"model_{epoch}_{_loss}.pt"))
    wandb.finish()


if __name__ == "__main__":
    main()