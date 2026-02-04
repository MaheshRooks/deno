import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ------------------------------
# Emotion Tone Instructions
# ------------------------------
emotion_tone_instructions = {
    "Angry": "Let's break this down clearly and calmly. ",
    "Happy": "Let's explore this exciting topic together! ",
    "Sad": "We'll take it slowly with a simple explanation. ",
    "Fearful": "No rush, we'll go through this step-by-step. ",
    "Neutral": "No change",
    "Surprised": "This sounds exciting! Let's dive into it! ",
}

# ------------------------------
# BiLSTM Model Definition
# ------------------------------
class BiLSTMEmotionModel(nn.Module):
    def __init__(self, input_size=2048, hidden_size=128, num_classes=5):  # Adjusted to match saved model
        super(BiLSTMEmotionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.bn(out)
        out = self.fc2(out)
        return out

# ------------------------------
# Load Trained Model
# ------------------------------
def load_emotion_model(model_path="emotion_bilstm_model.pt"):
    model = BiLSTMEmotionModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ------------------------------
# Extract CNN Features for LSTM
# ------------------------------
def extract_features(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    cnn = resnet50(weights=ResNet50_Weights.DEFAULT)  # ✅ Updated for deprecation
    cnn = nn.Sequential(*list(cnn.children())[:-1])
    cnn.eval()

    with torch.no_grad():
        features = cnn(img_tensor)
        features = features.view(1, 1, 2048)

    return features

# ------------------------------
# Predict Emotion
# ------------------------------
def predict_emotion(model, image_path):
    input_tensor = extract_features(image_path)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = output.argmax(dim=1).item()

    class_names = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised']
    return class_names[predicted_idx]

# ------------------------------
# Modify Prompt Based on Emotion
# ------------------------------
def modify_query_with_emotion(emotion, query):
    tone = emotion_tone_instructions.get(emotion, "")
    return tone + query if tone != "No change" else query

# ------------------------------
# Generate Explanation using GPT-2
# ------------------------------
def generate_explanation(modified_prompt, max_length=100):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # ✅ Fix missing pad_token issue
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer.encode(modified_prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(modified_prompt):].strip()

# ------------------------------
# Main Pipeline
# ------------------------------
def emotion_aware_explanation(image_path, user_query):
    model_path = "emotion_bilstm_model.pt"
    if not os.path.exists(model_path):
        print("❌ Model not found. Train and save it as 'emotion_bilstm_model.pt'.")
        return

    try:
        model = load_emotion_model(model_path)
        emotion = predict_emotion(model, image_path)
        tone_instruction = emotion_tone_instructions.get(emotion, "")

        modified_prompt = modify_query_with_emotion(emotion, user_query)
        explanation = generate_explanation(modified_prompt)

        # Show Image and Results
        img = Image.open(image_path).convert("RGB")
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
        full_text = (
            f"Emotion: {emotion}\n"
            f"Tone: {tone_instruction.strip()}\n\n"
            f"GPT-2 Response:\n{explanation}"
        )
        plt.figtext(0.5, -0.05, full_text, wrap=True, ha="center", fontsize=12)
        plt.tight_layout()
        plt.show()

        # Console Log
        print("\n--- Workflow Summary ---")
        print(f"User Query: {user_query}")
        print(f"Detected Emotion: {emotion}")
        print(f"Applied Rule: {tone_instruction}")
        print(f"GPT-2 Output: {explanation}")

    except Exception as e:
        print(f"❌ Error during processing: {e}")

# ------------------------------
# Run Example
# ------------------------------
if __name__ == "__main__":
    image_path = r"D:\Mahes\facial emotion\mahesh\mahesh\archive\train\Sad\0a0e635c934404c6b79feebfca3458762786381d9b9a04105d85dab7.jpg"
    user_query = "Write pandas code to read a CSV file and print the first 5 rows."
    emotion_aware_explanation(image_path, user_query)
