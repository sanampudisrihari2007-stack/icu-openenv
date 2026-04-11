import numpy as np
import torch
import torch.nn.functional as F
import json
import sys

def clamp_scores(scores, epsilon=1e-7):
    scores = np.asarray(scores)
    scores = np.clip(scores, epsilon, 1 - epsilon)
    return scores

def load_model(model_path):
    model = torch.nn.Linear(10, 1)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def predict_and_score(model, input_data, task_type='binary'):
    with torch.no_grad():
        outputs = model(input_data)

    if task_type == 'binary':
        scores = F.sigmoid(outputs).squeeze().numpy()
    elif task_type == 'multi':
        scores = F.softmax(outputs, dim=-1).numpy()
    else:
        scores = outputs.squeeze().numpy()

    valid_scores = clamp_scores(scores)

    output = {
        'scores': valid_scores.tolist()
    }
    
    return output

def main():
    model_path = 'model.pth'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    model = load_model(model_path)
    
    input_data = torch.randn(1, 10)
    
    results = predict_and_score(model, input_data, task_type='binary')
    
    with open('predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Inference completed.")

if __name__ == '__main__':
    main()