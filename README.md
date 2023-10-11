# AI_LinkedInRecommender
This Python script is designed for calculating and recommending users on LinkedIn based on their skills and a desired set of skills. It uses TensorFlow to create skill embeddings and compute user skill similarities. The script loads user data from a JSON file, extracts unique skills, and constructs a skill vocabulary. It then calculates and displays the top-K users who are most similar to a given set of skills.

## Key Components:
1. Utilizes TensorFlow for user skill similarity calculations.
2. Loads user data from a JSON file and extracts user URLs and skills.
3. Constructs a skill vocabulary for all unique skills in the dataset.
4. Computes skill-based user similarities.
5. Allows users to specify a set of desired skills for recommendations.

## To use the script:
1. Prepare your user data in a JSON file.
2. Define your desired set of skills.
3. Run the script to find and display the top-K users with the highest similarity scores for the specified skill set.

*The web scraping functionality was removed from this code due to concerns that it could be potentially misused for spamming purposes. I want to ensure the responsible and ethical use of the code and data, and therefore, decided to focus solely on the core recommendation system.* 
