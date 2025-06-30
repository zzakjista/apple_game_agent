
if __name__ == "__main__":
    from train.train import train_with_gui, train    
    from game.match_with_ai import match_with_ai

    # train_with_gui(episodes=10, max_steps=200, save_path1='result/agent1_final.pt', save_path2='result/agent2_final.pt')
    # train(episodes=10, max_steps=200, save_path1='result/agent1_final.pt', save_path2='result/agent2_final.pt')
    match_with_ai()