import tensorflow as tf
import numpy as np
import json

# Load user data from the JSON file
with open(r"C:\linkedin_data.json", "r") as json_file:
    user_data = json.load(json_file)

# Extract user URLs and skills from the loaded JSON data
user_urls = [user["user_id"] for user in user_data]
user_skills = [user["skills"] for user in user_data]

# Extract unique skills from the dataset
unique_skills = list(set(skill for user in user_data for skill in user["skills"]))

# Create a vocabulary for all unique skills
skill_vocabulary = tf.keras.layers.StringLookup(
    input_shape=(None,), mask_token=None, num_oov_indices=1, vocabulary=unique_skills
)

# Create embedding layers for user and skill embeddings
user_embedding = tf.keras.layers.Embedding(input_dim=len(user_urls), output_dim=32)
skill_embedding = tf.keras.layers.Embedding(input_dim=len(skill_vocabulary.get_vocabulary()), output_dim=32)

# Define the user similarity model
class UserSimilarityModel(tf.keras.Model):
    def __init__(self, user_embedding, skill_embedding):
        super().__init__()
        self.user_embedding = user_embedding
        self.skill_embedding = skill_embedding

    def call(self, inputs):
        user_indices, query_skills = inputs
        user_embeddings = self.user_embedding(user_indices)
        skill_embeddings = self.skill_embedding(query_skills)
        return user_embeddings, skill_embeddings

# Create an instance of the user similarity model
model = UserSimilarityModel(user_embedding, skill_embedding)

# Define the desired set of skills
desired_skills = [["Python", "Machine Learning", "Keras", "Data Visualization"]]

# Convert the desired skills to a tensor using the skill vocabulary
desired_skills = skill_vocabulary(desired_skills)

# Calculate similarities between the desired set of skills and each user's skills
user_similarities = []
for skills in user_skills:
    skills_tensor = skill_vocabulary(tf.constant([skills], dtype=tf.string))
    user_embeddings, skill_embeddings = model((tf.constant([0]), skills_tensor))  # Assuming query user has index 0
    similarity = tf.reduce_sum(user_embeddings * skill_embeddings)
    user_similarities.append(similarity.numpy())

# Find the top-K similar users for the desired set of skills
k = 3  # Number of top similar users to retrieve
top_k_user_indices = np.argsort(-np.array(user_similarities))[:k]

# Display the top-K recommended users with similarity scores for the desired skills
print("Recommended Users for the Desired Set of Skills:")
for user_index in top_k_user_indices:
    similar_user_url = user_urls[user_index]
    similarity_score = user_similarities[user_index]
    print(f"User URL: {similar_user_url}, Similarity Score: {similarity_score}")

