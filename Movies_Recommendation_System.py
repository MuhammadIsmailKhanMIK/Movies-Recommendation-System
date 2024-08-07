from tkinter import END, FLAT, WORD, Button, Entry, Label, Text, Tk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from tkinter import *
from tkinter import messagebox

# Load the movies data
try:
    movies = pd.read_csv('movies.csv')
except FileNotFoundError:
    messagebox.showerror("File Error", "The file 'movies.csv' was not found.")
    exit()

# Check if required columns are present
required_columns = ['title', 'genres']
if not all(column in movies.columns for column in required_columns):
    messagebox.showerror("Data Error", "Required columns are missing in the dataset.")
    exit()

# Preprocess: Convert genres to lowercase and fill NaN values
movies['genres'] = movies['genres'].fillna('').str.lower()

# Use TF-IDF Vectorizer to convert genres into vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Reduce dimensions with a number of components <= number of features
n_features = tfidf_matrix.shape[1]
n_components = min(20, n_features)  # Using 20 components, or less if there are fewer features
svd = TruncatedSVD(n_components=n_components)
tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)

# Create a Series with movie titles as index and indices as values
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Function to recommend movies
def recommend_movies(title, tfidf_matrix_reduced=tfidf_matrix_reduced, movies=movies):
    if title not in indices:
        return []
    idx = indices[title]
    # Compute cosine similarity for the given movie only
    cosine_similarities = linear_kernel(tfidf_matrix_reduced[idx:idx+1], tfidf_matrix_reduced).flatten()
    similar_indices = cosine_similarities.argsort()[::-1][1:11]  # Top 10 recommendations excluding the movie itself
    return movies['title'].iloc[similar_indices].tolist()

# Function to show recommendations
def show_recommendations():
    title = entry.get().strip()
    if not title:
        messagebox.showwarning("Input Error", "Please enter a movie title.")
        return
    recommendations = recommend_movies(title)
    recommendations_text.delete(1.0, END)  # Clear previous recommendations
    if not recommendations:
        recommendations_text.insert(END, "No recommendations found for the given title.")
    else:
        recommendations_text.insert(END, "Recommendations for '{}':\n\n".format(title), 'bold')
        for rec in recommendations:
            recommendations_text.insert(END, rec + '\n')

# Create the main window
root = Tk()
root.title("Professional Movie Recommendation System")
root.geometry("600x450")
root.resizable(False, False)
root.configure(bg='#2e2e2e')

# Set the font and styles
font_family = "Segoe UI"
header_font = (font_family, 18, 'bold', 'underline')
subheader_font = (font_family, 14, 'bold', 'underline')
normal_font = (font_family, 12)
bold_font = (font_family, 12, 'bold')
btn_font = (font_family, 12, 'bold')
bg_color = '#2e2e2e'
text_color = '#ffffff'
button_color = '#0052cc'
button_text_color = '#ffffff'
entry_bg_color = '#404040'
entry_text_color = '#ffffff'
text_bg_color = '#404040'
text_text_color = '#ffffff'

# Main Title Label
Label(root, text="Movie Recommendation System", font=header_font, bg=bg_color, fg=text_color).pack(pady=20)

# Input Section Heading
Label(root, text="Enter Movie Title", font=subheader_font, bg=bg_color, fg=text_color).pack()

# Input Label and Entry
Label(root, text="Enter a movie title:", font=normal_font, bg=bg_color, fg=text_color).pack()
entry = Entry(root, width=40, font=normal_font, bg=entry_bg_color, fg=entry_text_color)
entry.pack(pady=5)

# Button to get recommendations
Button(root, text="Get Recommendations", font=btn_font, command=show_recommendations, bg=button_color, fg=button_text_color, relief=FLAT).pack(pady=15)

# Text widget to display recommendations
recommendations_text = Text(root, width=70, height=10, font=normal_font, bg=text_bg_color, fg=text_text_color, wrap=WORD)
recommendations_text.pack(pady=10)
recommendations_text.tag_configure('bold', font=(font_family, 12, 'bold'))

# Start the GUI loop
root.mainloop()