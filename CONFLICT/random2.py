import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

csv_path = r"C:\Users\ADMIN\Downloads\HEALTHSYNC\Conflict\drugs_for_common_treatments.csv"
drugs_data = pd.read_csv(csv_path)

user_data_path = r"C:\Users\ADMIN\Downloads\HEALTHSYNC\Conflict\User.csv"
user_data = pd.read_csv(user_data_path)

# Load substitute data
substitute_data_path = r"C:\Users\ADMIN\Downloads\HEALTHSYNC\Conflict\Substitue.csv"
substitute_data = pd.read_csv(substitute_data_path)

# Load side effects data
side_effects_data_path = r"C:\Users\ADMIN\Downloads\HEALTHSYNC\Conflict\drugs_side_effects_drugs_com.csv"
side_effects_data = pd.read_csv(side_effects_data_path)


def get_top_drugs(disease, drugs_data):
    # Drop NaN values from 'medical_condition_description' column
    drugs_data_cleaned = drugs_data.dropna(subset=['medical_condition_description'])

    vectorizer = TfidfVectorizer()
    all_text = list(drugs_data_cleaned['medical_condition_description']) + [disease]
    vectors = vectorizer.fit_transform(all_text)

    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1])

    most_similar_index = similarity_scores.argmax()

    relevant_disease = drugs_data_cleaned.iloc[most_similar_index]

    relevant_drugs = drugs_data_cleaned[drugs_data_cleaned['medical_condition'] == relevant_disease['medical_condition']]
    top_drugs = relevant_drugs.sort_values(by='rating', ascending=False).head(3)

    response = "\nTop Drugs:\n"
    for _, drug_info in top_drugs.iterrows():
        response += f"Drug: {drug_info['drug_name']}, Rating: {drug_info['rating']}\n"

    return response, top_drugs


def get_user_record(username, password, user_data):
    return user_data[(user_data['username'] == username) & (user_data['password'] == password)]


def update_user_record(username, password, diagnosis, user_data):
    user_index = get_user_record(username, password, user_data).index[0]

    # Check for available slots (old, new_1, new_2)
    available_slots = ['old_1', 'old_2', 'old_3', 'new_1', 'new_2']
    for slot in available_slots:
        if pd.isnull(user_data.loc[user_index, slot]):
            user_data.at[user_index, slot] = diagnosis
            user_data.to_csv(user_data_path, index=False)
            return slot

    return None  # All slots are full


def extract_drugs_from_string(drugs_string):
    # Assuming the string is in the format: Drug: [drug_name], Rating: [rating]\n
    drugs_list = drugs_string.split('\n')[1:-1]
    drugs_info = [info.split(', ') for info in drugs_list]
    drugs_df = pd.DataFrame(drugs_info, columns=['Drug', 'Rating'])
    return drugs_df


def substitute_top_drugs(matching_drugs, substitute_data):
    substituted_drugs = {}
    drugs_df = extract_drugs_from_string(matching_drugs)
    new_drug_info = pd.DataFrame({'Drug': ['doxycycline'], 'Rating': [9.0]})  # Add your desired rating
    drugs_df = pd.concat([drugs_df, new_drug_info], ignore_index=True)
    drugs_df = drugs_df.drop(0)

    for _, drug_info in drugs_df.iterrows():
        drug_name = drug_info['Drug']
        side_effect_info = side_effects_data[side_effects_data['drug_name'] == drug_name]

        if not side_effect_info.empty:
            side_effects = side_effect_info.iloc[0]['side_effects']
            substituted_drugs[drug_name] = side_effects

            # Get substitute drug from substitute_data
            substitute_drug_info = substitute_data[substitute_data['drug_name'] == drug_name]
            if not substitute_drug_info.empty:
                substitute_drug_name = substitute_drug_info.iloc[0]['substitute']
                substituted_drugs[substitute_drug_name] = side_effects

    return substituted_drugs


def main():
    global user_data
    all_slots_filled = True

    print("Disease Diagnosis App")

    print("Are you a new or existing user?")
    user_type = input("Select user type: (New/Existing)").capitalize()

    if user_type == "New":
        # Get user information
        username = input("Enter a username:")
        password = input("Enter a password:")

        # Create a new user record
        new_user = pd.DataFrame({'username': [username],
                                 'password': [password],
                                 'side_effect1': [""],
                                 'side_effect2': [""],
                                 'side_effect3': [""],
                                 'side_effect4': [""],
                                 'side_effect5': [""],
                                 'old_1': [""],
                                 'old_2': [""],
                                 'old_3': [""],
                                 'new_1': [""],
                                 'new_2': [""]})

        # Append the new user record to the user_data DataFrame
        user_data = pd.concat([user_data, new_user], ignore_index=True)
        user_data.to_csv(user_data_path, index=False)

        print("New user created successfully. Please log in with your credentials.")
    elif user_type == "Existing":
        # Get user login information
        username = input("Enter your username:")
        password = input("Enter your password:")

        # Check if the user exists
        user_record = get_user_record(username, password, user_data)

        if not user_record.empty:
            user_index = user_record.index[0]

            # Check for available slots
            available_slot = None
            if all_slots_filled:
                available_slot = update_user_record(username, password, None, user_data)
            else:
                for i in range(1, 4):
                    old_col = f'old_{i}'
                    if pd.isnull(user_data.loc[user_index, old_col]):
                        available_slot = old_col
                        break
                else:
                    all_slots_filled = True

            if available_slot:
                user_disease = input("Enter your Disease:")
                if user_disease:
                    result_drugs, top_drugs = get_top_drugs(user_disease, drugs_data)
                    print(top_drugs)

                    # Update the user record with symptoms, diagnosis, and side effects
                    print(f"Top Drugs:\n{result_drugs}")
                    update_user_record(username, password, user_disease, user_data)
                    drug_names = [drug_info.drug_name for drug_info in top_drugs.itertuples()]
                    print(drug_names)
                    drug_side_effects = {}

                    # Search for side effects of each drug
                    for drug_name in drug_names:
                        side_effect_info = side_effects_data[side_effects_data['drug_name'] == drug_name]
                        if not side_effect_info.empty:
                            side_effects = side_effect_info.iloc[0]['side_effects']
                            side_effects_set = set(side_effects.lower().split(', '))

                            # Store each word of side effects separately in the effects set
                            effects_set = set()
                            for effect in side_effects_set:
                                effects_set.update(effect.split())

                            drug_side_effects[drug_name] = effects_set

                            # Print side effects for verification
                            #print(f"Side effects of {drug_name}: {effects_set}")
                        else:
                            print(f"No side effects found for {drug_name}")

                    # Display side effects
                    for drug, effects in drug_side_effects.items():
                        print(f"Drug: {drug}, Side Effects: {effects}")

                    # Check if any of the patient's history diseases match with the side effects set
                    for i in range(1, 4):
                        old_col = f'old_{i}'
                        if pd.notnull(user_data.loc[user_index, old_col]):
                            history_disease = user_data.loc[user_index, old_col].lower()
                            for drug, effects in drug_side_effects.items():
                                if history_disease in effects:
                                    print(f"Conflict Detected:")
                                    print(f"Patient History Disease: {history_disease}")
                                    print(f"Prescribed Drug: {drug}")
                                    print(f"Words Present in Side Effects: {effects.intersection(history_disease.split())}")


                                    # Suggest a substitute drug if available
                                    substitute_drugs = []
                                    for j in range(5):  # Assuming substitute drugs are named substitute0 to substitute4
                                                substitute_col = f'substitute{j}'
                                                substitute_drug_info = substitute_data[substitute_data['drug_name'] == drug]
                                                if not substitute_drug_info.empty:
                                                       substitute_drug = substitute_drug_info.iloc[0][substitute_col]
                                                       substitute_drugs.append(substitute_drug)


                                    

                # Select the top three substitutes based on some criteria (e.g., rating)
                                    top_substitutes = substitute_drugs[:3]

                                    valid_substitutes = []
                                    for substitute in top_substitutes:
                                                        substitute_effects = side_effects_data[side_effects_data['drug_name'] == substitute]
                                                        if not substitute_effects.empty:
                                                            substitute_side_effects = set(substitute_effects.iloc[0]['side_effects'].lower().split(', '))
                                                            common_effects = substitute_side_effects.intersection(history_disease.split())
                                                            if not common_effects:
                                                                        valid_substitutes.append(substitute)
                                                            else:
                                                                        print(f"Substitute Drug: {substitute}")
                                                                        print(f"Side Effects Present in Patient's History: {', '.join(common_effects)}")


                                    print("Top  Substitutes:")
                                    for substitute in valid_substitutes:
                                         print(substitute)
                                    
                                    

if __name__ == "__main__":
    main()



