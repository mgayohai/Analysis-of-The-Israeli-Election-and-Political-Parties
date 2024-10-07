import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from wordcloud import WordCloud

hebrew_stopwords = [
    "שכן", "אותה", "קרוב", "ו", "ה", "האם", "מכיוון", "מבין", "עת", "תחת", "נכון", "לאור", "עם", "נגד",
    "רוב", "במקום", "במשך", "אשר", "כל", "הרבה", "מתחת", "מדי", "בזכות", "סביב", "אחרי", "לעומת",
    "כ", "לה", "הן", "כמה", "גם", "ללא", "אצל", "כלל", "מאת", "מרבית", "אף", "יותר", "לו", "לא",
    "מן", "כיוון", "יש", "בקרב", "למען", "את", "למשל", "עצמם", "תוך", "לשם", "כי", "מתוך", "בשל",
    "מעבר", "עוד", "מטעם", "פי", "אלא", "עצמה", "מ", "משום", "פחות", "אחר", "לבין", "אלה",
    "לצד", "זו", "יחד", "עבור", "בעוד", "לפני", "הללו", "מה", "הוא", "כן", "לפי", "כדי", "היא",
    "כנגד", "שאר", "עבר", "אבל", "בתוך", "אי", "כגון", "עקב", "בפני", "בעקבות", "בתור", "זה",
    "בגלל", "מי", "קודם", "מעל", "של", "אך", "לאחר", "מנת", "א", "בין", "כמו", "אולם", "ב",
    "בניגוד", "לגבי", "החל", "כלפי", "מספר", "בה", "דרך", "ל", "מאז", "או", "מפני", "על",
    "לקראת", "אותם", "לאורך", "הרי", "אני", "הם", "אלו", "כך", "מעט", "רק", "מהם", "למרות", "אותו",
    "מול", "מאחר", "אם", "מצד", "ליד", "עצמו", "ידי", "זהו", "לידי", "בידי", "זאת", "באמצעות",
    "ככל", "עד", "כאשר", "אל", "מאשר", "כפי", "כלומר", "אין", "היה", "היו", "אנו"
]


def read_file(filepath):
    """Read and return the content of a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        print(f"Encoding error while reading {filepath}. Skipping this file.")
        return ""


def is_valid_term(term):
    """Check if a term is valid (Hebrew letters only, after removing quotes, apostrophes, and hyphens)."""
    # Remove unwanted characters: ' " -
    cleaned_term = re.sub(r"['\"-]", "", term)

    # Check if the cleaned term contains only Hebrew letters
    return re.match(r"^[\u0590-\u05FF]+$", cleaned_term)  # Hebrew letters only


def get_party_and_year(filename):
    """Helper to extract party and year from filename."""
    parts = filename.split()  # Adjust if filenames use a different format
    party = " ".join(parts[:-1])  # All parts except the last are assumed to be the party name
    year = parts[-1].split('.')[0]  # Last part (before extension) is assumed to be the year
    return party, year


def file_generator(directory):
    """Yield file content one by one, cleaning unwanted characters."""
    for filepath in [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]:
        # Read the content of the file
        content = read_file(filepath)
        # Remove digits, quotes, and non-Hebrew characters (including English text)
        content = re.sub(r'[^\u0590-\u05FF\s]', '', content)  # Keep only Hebrew letters and spaces
        yield content, filepath  # Yield cleaned content and filepath


def compute_tfidf_for_files(directory, stop_words=None, option=3):
    """Compute the TF-IDF scores for all text files in a directory, based on the selected option."""

    # Initialize dictionaries to store party, year, and file info
    party_platforms = defaultdict(list)
    year_platforms = defaultdict(list)
    file_info = {}  # Stores (party, year) for each file

    # Use the generator to load file content
    for content, filepath in file_generator(directory):
        party, year = get_party_and_year(os.path.basename(filepath))
        party_platforms[party].append((content, year))
        year_platforms[year].append(content)
        file_info[filepath] = (party, year)

    # Handle the user's chosen option
    documents, doc_names = [], []
    if option == 1:
        for party, platforms in party_platforms.items():
            last_platform = platforms[-1]
            documents.append(last_platform[0])
            doc_names.append(f"{party} ({last_platform[1]})")
    elif option == 2:
        documents = [" ".join([platform[0] for platform in platforms]) for platforms in party_platforms.values()]
        doc_names = [f"{party} ({', '.join([platform[1] for platform in platforms])})" for party, platforms in
                     party_platforms.items()]
    elif option == 4:
        documents = [" ".join(platforms) for platforms in year_platforms.values()]
        doc_names = [f"{year}" for year in year_platforms]
    else:
        # Option 3: Each platform as a separate document
        for content, filepath in file_generator(directory):
            documents.append(content)
            doc_names.append(f"{file_info[filepath][0]} ({file_info[filepath][1]})")

    # Create a TfidfVectorizer with Hebrew stop words (if provided)
    vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=2)

    # Fit and transform the documents into a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)

    return tfidf_matrix, vectorizer, doc_names


def display_top_tfidf_terms(tfidf_matrix, feature_names, doc_names, top_n=3):
    """Display top N TF-IDF terms for each document."""
    for i, doc_name in enumerate(doc_names):
        tfidf_scores = tfidf_matrix[i].toarray()[0]
        # Filter and display only valid terms (Hebrew letters)
        valid_scores = {term: tfidf_scores[idx] for idx, term in enumerate(feature_names) if is_valid_term(term)}
        valid_scores = np.array(list(valid_scores.values()))  # Convert to NumPy array for sorting
        print_top_tfidf_scores(valid_scores, feature_names, doc_name, top_n=top_n)


# Function to subtract two vectors
def subtract_vectors(tfidf_matrix, file1_idx, file2_idx):
    """Subtract the TF-IDF vector of file2 from file1."""
    return tfidf_matrix[file1_idx].toarray() - tfidf_matrix[file2_idx].toarray()


# Find the closest files to the resulting vector
def find_closest_files(result_vector, tfidf_matrix, doc_names, top_n=3):
    """Find the top_n files closest to the resulting vector."""
    similarities = cosine_similarity(result_vector, tfidf_matrix).flatten()

    # Get the indices of the top_n most similar documents
    closest_indices = np.argsort(similarities)[-top_n:][::-1]  # Sort and get top_n indices
    closest_files = [doc_names[idx] for idx in closest_indices]
    closest_scores = [similarities[idx] for idx in closest_indices]

    return list(zip(closest_files, closest_scores))


# Function to get the vector for a specific term
def get_term_vector(term, vectorizer, tfidf_matrix):
    """Get the vector for a specific term."""
    try:
        term_idx = vectorizer.vocabulary_[term.lower()]
        term_vector = tfidf_matrix[:, term_idx].toarray()
        return term_vector
    except KeyError:
        print(f"Term '{term}' not found in the vocabulary.")
        return None


def vector_arithmetic_and_similarity(vectorizer, tfidf_matrix, feature_names):
    """Perform vector arithmetic (A + B - C) in a loop with options for user interaction."""

    def get_term_vector(term):
        """Helper function to get the vector for a specific term."""
        try:
            term_idx = vectorizer.vocabulary_[term.lower()]
            return tfidf_matrix[:, term_idx].toarray()
        except KeyError:
            print(f"Term '{term}' not found in the vocabulary.")
            return None

    def get_user_term():
        """Helper function to get a valid term input from the user."""
        return get_valid_input("Enter a term (in Hebrew): ", vectorizer.vocabulary_, False,
                               "The term Was not found, Try again!")

    result_vector = None

    while True:
        # Ask for term and whether to add or subtract it
        if result_vector is None:
            term = get_user_term()  # Get initial term from the user
            result_vector = get_term_vector(term)
        else:
            operation = get_valid_input("Do you want to add (+) or subtract (-) a term? ", ['+', '-'])
            term = get_user_term()

            term_vector = get_term_vector(term)

            if operation == '+':
                result_vector = np.add(result_vector, term_vector)
            elif operation == '-':
                result_vector = np.subtract(result_vector, term_vector)

        # Calculate cosine similarity with all documents in tfidf_matrix
        similarities = cosine_similarity(result_vector.T, tfidf_matrix.T).flatten()

        # Get the indices of the top 3 most similar terms
        closest_term_indices = np.argsort(similarities)[-3:][::-1]  # Sort in descending order

        # Display the 3 closest terms and their similarity scores
        print(f"\nThe 3 terms closest to your current vector result are:")
        for idx in closest_term_indices[:3]:
            print(f"Term: {feature_names[idx]}, Similarity Score: {similarities[idx]:.4f}")

        # Ask if the user wants to continue with the result or start over
        action = get_valid_input("Do you want to continue with this result (c), reset (r), or exit (e)? ",
                                 ['c', 'r', 'e'])

        if action == 'r':
            result_vector = None  # Reset result
        elif action == 'e':
            print("Exiting vector arithmetic.")
            break  # Exit the loop
        else:
            continue  # Continue adding/subtracting terms


def print_top_tfidf_scores(tfidf_scores, feature_names, doc_name, top_n=15):
    """Helper function to print top N TF-IDF scores."""
    sorted_indices = np.argsort(tfidf_scores)[::-1][:top_n]  # Sort scores in descending order
    print(f"Top {top_n} TF-IDF terms for {doc_name}:")
    for idx in sorted_indices:
        print(f"{feature_names[idx]}: {tfidf_scores[idx]}")


def generate_word_cloud_single(tfidf_scores, feature_names, doc_name):
    """Generate and display a word cloud for a document based on TF-IDF scores."""
    word_scores = {feature_names[idx][::-1]: tfidf_scores[idx] for idx in range(len(tfidf_scores)) if
                   is_valid_term(feature_names[idx]) and tfidf_scores[idx] > 0}  # Only include positive scores
    if not word_scores:
        print("No valid words to generate a word cloud.")
        return  # Exit if there are no valid words

    font_path = 'Alef-Regular.ttf'  # Ensure this path is correct
    wordcloud = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color='white'
    ).generate_from_frequencies(word_scores)

    # Plot the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"Word Cloud for {doc_name}", fontsize=16)
    plt.axis("off")
    plt.show()


def generate_word_clouds_side_by_side(tfidf_scores1, doc_name1, tfidf_scores2, doc_name2, feature_names):
    """Generate and display two word clouds side by side for comparison."""
    # Create dictionaries of terms and their corresponding TF-IDF scores for both documents
    word_scores1 = {feature_names[idx][::-1]: tfidf_scores1[idx] for idx in range(len(tfidf_scores1)) if
                    is_valid_term(feature_names[idx])}
    word_scores2 = {feature_names[idx][::-1]: tfidf_scores2[idx] for idx in range(len(tfidf_scores2)) if
                    is_valid_term(feature_names[idx])}

    font_path = 'Alef-Regular.ttf'  # Ensure this path is correct
    wordcloud1 = WordCloud(font_path=font_path, width=800, height=400,
                           background_color='white').generate_from_frequencies(word_scores1)
    wordcloud2 = WordCloud(font_path=font_path, width=800, height=400,
                           background_color='white').generate_from_frequencies(word_scores2)

    # Create a side-by-side plot
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Word cloud for the first document
    axs[0].imshow(wordcloud1, interpolation="bilinear")
    axs[0].set_title(f"Word Cloud for {doc_name1}", fontsize=16)
    axs[0].axis("off")

    # Word cloud for the second document
    axs[1].imshow(wordcloud2, interpolation="bilinear")
    axs[1].set_title(f"Word Cloud for {doc_name2}", fontsize=16)
    axs[1].axis("off")

    # Show the plot
    plt.show()


def word_cloud_generation(tfidf_matrix, doc_names, vectorizer):
    """Allow the user to generate word clouds for selected files in a loop."""
    while True:
        print("\nAvailable documents for word cloud generation:")
        for i, doc_name in enumerate(doc_names):
            print(f"{i + 1}. {doc_name}")

        doc_names_str_range = list(map(str, range(1, len(doc_names) + 1)))

        # Let the user choose two files for word cloud generation
        file1_idx = int(
            get_valid_input("Select the index of the first file for the word cloud: ", doc_names_str_range)) - 1
        file2_idx = int(
            get_valid_input("Select the index of the second file for the word cloud: ", doc_names_str_range)) - 1

        # Get the TF-IDF scores for the selected files
        tfidf_scores_file1 = tfidf_matrix[file1_idx].toarray()[0]
        tfidf_scores_file2 = tfidf_matrix[file2_idx].toarray()[0]

        # Generate side-by-side word clouds for the selected files
        generate_word_clouds_side_by_side(tfidf_scores_file1, doc_names[file1_idx], tfidf_scores_file2,
                                          doc_names[file2_idx], vectorizer.get_feature_names_out())

        # Ask if the user wants to continue or exit
        action = get_valid_input("Do you want to generate more word clouds (c) or exit (e)? ", ['c', 'e'])

        if action == 'e':
            print("Exiting word cloud generation.")
            break  # Exit the loop


# Function to add vectors
def add_vectors(tfidf_matrix, file1_idx, file2_idx):
    """Add the TF-IDF vector of file2 to file1."""
    return tfidf_matrix[file1_idx].toarray() + tfidf_matrix[file2_idx].toarray()


def file_arithmetic(tfidf_matrix, doc_names):
    """Allow the user to perform file addition or subtraction in a loop with options for reset or continue."""
    prev_result_vector = None

    while True:
        result_vector = None
        # Display available documents
        print("\nAvailable documents:")
        for i, doc_name in enumerate(doc_names):
            print(f"{i + 1}. {doc_name}")

        # Get user input for file selection
        doc_names_str_range = list(map(str, range(1, len(doc_names) + 1)))

        if prev_result_vector is None:
            file_idx = int(get_valid_input("Select the index of the first file: ", doc_names_str_range)) - 1
            result_vector = tfidf_matrix[file_idx].toarray()
        else:
            operation = get_valid_input("Do you want to add (+) or subtract (-) another file? ", ['+', '-'])
            file_idx = int(get_valid_input("Select the index of the next file: ", doc_names_str_range)) - 1

            if operation == '+':
                result_vector = prev_result_vector + tfidf_matrix[file_idx].toarray()

            elif operation == '-':
                result_vector = prev_result_vector - tfidf_matrix[file_idx].toarray()

        # Find the closest files to the resulting vector
        closest_files_with_scores = find_closest_files(result_vector, tfidf_matrix, doc_names, top_n=3)

        # Display the top 3 closest matches
        print("\nThe top 3 files closest to the current result are:")
        for closest_file, score in closest_files_with_scores:
            print(f"'{closest_file}' with a similarity score of {score:.4f}")

        if prev_result_vector is not None:
            # Get the TF-IDF scores for the selected files
            tfidf_scores_file1 = prev_result_vector[0]
            tfidf_scores_file2 = result_vector[0]

            # Generate side-by-side word clouds for the selected files
            generate_word_clouds_side_by_side(tfidf_scores_file1, "prev res", tfidf_scores_file2,
                                              "res", vectorizer.get_feature_names_out())

        # Ask if the user wants to continue, reset, or exit
        action = get_valid_input("Do you want to continue (c), reset (r), or exit (e)? ", ['c', 'r', 'e'])

        if action == 'r':
            prev_result_vector = None  # Reset result
        elif action == 'e':
            print("Exiting file arithmetic.")
            break  # Exit the loop
        else:
            prev_result_vector = result_vector


def get_valid_input(prompt, valid_choices, upon_mistake_print_valids_=True,
                    err_msg="Invalid choice. "):
    """Get valid input from the user, ensuring it is in the valid choices."""
    while True:
        choice = input(prompt + "\n").strip()
        if choice.lower() in valid_choices:
            return choice.lower()
        else:
            print(err_msg)
            if upon_mistake_print_valids_:
                print(f"Please enter one of {valid_choices}.")


# Function to perform clustering
def perform_clustering(tfidf_matrix, n_clusters):
    """Perform K-Means clustering on the TF-IDF matrix."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)

    return kmeans.labels_


def plot_clusters(labels, tfidf_matrix, doc_names):
    """Plot the clusters."""
    plt.figure(figsize=(12, 6))
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_docs = [doc_names[i] for i in range(len(doc_names)) if labels[i] == label]
        plt.scatter([label] * len(cluster_docs), cluster_docs, label=f'Cluster {label}')

    plt.xlabel("Cluster")
    plt.ylabel("Documents")
    plt.title("Clusters of Political Platforms")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()  # Automatically adjust the padding
    plt.show()


# try 2
def plot_clusters_2d(labels, tfidf_matrix, doc_names):
    """Perform PCA and plot clusters in 2D space."""
    """Perform PCA to reduce the TF-IDF matrix to 2D and plot clusters with annotations."""

    # Perform PCA to reduce the TF-IDF matrix to 2D for visualization
    pca = PCA(n_components=2)
    reduced_matrix = pca.fit_transform(tfidf_matrix.toarray())  # Convert to dense for PCA

    plt.figure(figsize=(10, 6))

    # Plot the reduced 2D matrix with cluster labels as colors
    scatter = plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], c=labels, cmap='viridis', s=100)

    # Annotate each point with its corresponding document name
    for i, doc_name in enumerate(doc_names):
        plt.annotate(doc_name, (reduced_matrix[i, 0], reduced_matrix[i, 1]), fontsize=8, ha='right')

    plt.title("2D PCA Visualization of Clusters")
    plt.colorbar(scatter, label="Cluster Labels")  # Add a colorbar to indicate cluster labels
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    plt.show()


def perform_clustering_and_display(tfidf_matrix, doc_names):
    """Perform clustering on the TF-IDF matrix and display results."""
    n_clusters = int(get_valid_input("Enter the number of clusters you want to create: ", list(map(str, range(1, 15)))))

    # Perform clustering
    labels = perform_clustering(tfidf_matrix, n_clusters)

    # Print cluster assignments
    for i, doc_name in enumerate(doc_names):
        print(f"{doc_name} is in cluster {labels[i]}")

    # Optionally plot clusters
    plot_clusters(labels, tfidf_matrix, doc_names)


directory = 'MAZA/txt files'  # directory path to party platforms txt files

# Main program loop
while True:
    # Prompt the user for their choice
    print("How would you like to handle the party platforms?")
    print("1 - Use the last platform per party")
    print("2 - Union all platforms of the same party")
    print("3 - Leave it as it is (each platform separate)")
    print("4 - Union all platforms by year")

    user_choice = get_valid_input("Enter your choice (1, 2, 3, or 4): ", ['1', '2', '3', '4'])
    user_choice = int(user_choice)  # Convert the valid string choice to integer

    # Compute TF-IDF based on user's choice
    tfidf_matrix, vectorizer, doc_names = compute_tfidf_for_files(directory, stop_words=hebrew_stopwords,
                                                                  option=user_choice)

    # Ask if the user wants to receive top tfidf terms (3)
    perform_top_tfidf_for_files = get_valid_input("Do you want to receive top tfidf terms (3) of each file? (yes/no): ",
                                                  ['yes', 'no'])
    if perform_top_tfidf_for_files == 'yes':
        # Get the feature names (i.e., terms)
        feature_names = vectorizer.get_feature_names_out()
        # Display top 3 TF-IDF terms for each document
        display_top_tfidf_terms(tfidf_matrix, feature_names, doc_names, top_n=3)

    # Ask if the user wants to receive word cloud generation
    perform_word_cloud = get_valid_input("Do you want to generate word clouds comparing two files? (yes/no): ",
                                         ['yes', 'no'])
    if perform_word_cloud == 'yes':
        word_cloud_generation(tfidf_matrix, doc_names, vectorizer)

    # Ask if the user wants to perform vector arithmetic - terms
    perform_vector_arithmetic = get_valid_input(
        "Do you want to perform vector arithmetic (A + B - C) with terms? (yes/no): ",
        ['yes', 'no'])
    if perform_vector_arithmetic == 'yes':
        vector_arithmetic_and_similarity(vectorizer, tfidf_matrix, vectorizer.get_feature_names_out())

    # Ask if the user wants to perform vector arithmetic - files
    perform_file_arithmetic = get_valid_input(
        "Do you want to perform file arithmetic (add/subtract files)? (yes/no): ", ['yes', 'no'])
    if perform_file_arithmetic == 'yes':
        file_arithmetic(tfidf_matrix, doc_names)

    # Ask if the user wants to perform clustering
    # perform_clustering_option = get_valid_input("Do you want to perform clustering on the platforms? (yes/no): ",
    #                                             ['yes', 'no'])
    # if perform_clustering_option == 'yes':
    #     perform_clustering_and_display(tfidf_matrix, doc_names)

    restart_option = get_valid_input("Do you want to restart the process or exit? (r/e): ", ['r', 'e'])
    if restart_option == 'e':
        print("Exiting the program. Goodbye!")
        break
    print("let's start over! \n")
