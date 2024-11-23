# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout 

# %%
df = pd.read_csv('Sentiment/sentiment_analysis.csv')


def clean_text(text):

    
    # Remove non-alphabetical characters
    cleaned_text = ""
    for char in text:
        if char.isalpha() or char.isspace():
            cleaned_text += char
        else:
            cleaned_text += " "
    
    # Remove extra spaces
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text

# Apply cleaning to the text column
df['cleaned_text'] = df['text'].apply(clean_text)


df.head(30)

# %%
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])  # positive=2, neutral=1, negative=0

# 4. Split Data
x = df['cleaned_text']
y = df['sentiment_encoded']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

# Find maximum sequence length
maxlen = max(len(seq) for seq in train_sequences)

train_padded = pad_sequences(train_sequences, maxlen=maxlen, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=maxlen, padding='post')

# %%
input_size = np.max(train_padded) + 1
print(input_size)
     

# %%
model = Sequential([
    Embedding(input_dim=input_size, output_dim=100, input_length=maxlen),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # Output layer for 3 classes
])

# %%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 7. Train the Model
history = model.fit(train_padded, y_train, epochs=50, batch_size=32, validation_data=(test_padded, y_test))

# 8. Evaluate the Model
loss, accuracy = model.evaluate(test_padded, y_test)
print("Test Accuracy:", accuracy)

# 9. Predict on New Data
new_texts = ["This is an amazing day", "I'm not happy with this"]
new_texts_cleaned = [clean_text(text) for text in new_texts]
new_sequences = tokenizer.texts_to_sequences(new_texts_cleaned)
new_padded = pad_sequences(new_sequences, maxlen=maxlen, padding='post')

predictions = model.predict(new_padded)
predicted_labels = [label_encoder.inverse_transform([np.argmax(pred)])[0] for pred in predictions]
print("Predicted Labels:", predicted_labels)

# %%
new_texts = ["its a very pretty day", "i am angry and distrub today"]
new_texts_cleaned = [clean_text(text) for text in new_texts]
new_sequences = tokenizer.texts_to_sequences(new_texts_cleaned)
new_padded = pad_sequences(new_sequences, maxlen=maxlen, padding='post')

predictions = model.predict(new_padded)
predicted_labels = [label_encoder.inverse_transform([np.argmax(pred)]) for pred in predictions]
print("Predicted Labels:", predicted_labels)

