import pandas as pd
from num2words import num2words

df = pd.read_csv("profile.csv")

row = df.iloc[0:1]
print("Row: ", row)

# Basic Personal Information
name_2 = row["Name"][0]
print("Name 2: ", name_2)
age = row["Age"][0]
print("Age: ", age)
age_word = num2words(age)
print(age_word)
gender = row["Gender"][0]
print("Gender: ", gender)
if gender == "Female":
    gender_2 = "Woman"
elif gender == "Male":
    gender_2 = "Man"
elif gender == "Non-binary":
    gender_2 = gender
elif gender == "Prefer not to say":
    gender_2 = "Prefer not to say "
else:
    gender_2 = "Other"
print("Gender 2: ", gender_2)
live_in = row["Live in"][0]
print("Live in: ", live_in)

# Personal Relationships
marital_stat = row["Please indicate your marital status"][0]
print("Please indicate your marital status: ", marital_stat)
living_situ = row["Please indicate your living situation"][0]
print("Please indicate your living situation: ", living_situ)
imm_family = row["Would you like to tell us more about your immediate family"][0]
print("Would you like to tell us more about your immediate family: ", imm_family)
have_pets = row["Have you ever had pets"][0]
pets_info = row["If you are comfortable sharing, please tell us more about your pets."][0]
friends_info = row["Is there anything you would like the voice assistant to know about important people in your life (such as close friends or family members) to make conversations more personal"][0]

# Education and Work
ed_level = row["Please indicate your highest level of education"][0]
# had_job = row["Do you have any work experience"][0]
job_y = row["If you have worked, would you like to describe about your job or area of expertise"][0]
job_n = row["If you have not worked, would you like to share how was your life like during your 20s to 40s"][0]

# Preferences
typ_day = row["Can you tell us what a typical day is like for you"][0]
act_others = row["Please mention a few activities you would like to do with your family and friends."][0]
own_act = row["Please mention a few other activities that you would like to do on your own time"][0]
own_act_dislike = row["Please mention a few activities that you dislike or prefer not to do in your daily life"][0]

# Interaction Style
first_talk = row["What topics do you like to talk about when you meet a new person for the first time"][0]
int_style = row["What interaction style would you prefer the most when you are having a conversation"][0]

# Environment and Lifestyle and Technology Experience
tech_exp = row["How often do you use technology"][0]
tech_devices = row["What are the technological devices you use frequently in your life"][0]
va_use = row["Have you ever used voice assistants"][0]
va_use_y = True if va_use == "Yes" else False
va_type_y = row["Which voice assistant(s) have you used before"][0]

# Goals for the AI by the user
goals_y = row["For what purposes would you likely to use a conversational voice assistant like this"][0]


# Writing to text file
with open("demoprofile.txt", "a") as f:
    # Basic Personal Information
    f.write("Personal Information about the user:" + "\n")
    f.write("- My name is " + name_2 + "\n")
    f.write("- I am " + str(age) + " years old." + "\n")
    f.write("- I am " + age_word + " years old." + "\n")
    if gender not in ["Prefer not to say", "Other"]:
      f.write("- I am a " + gender + "\n")
      f.write("- I am a " + gender_2 + "\n")
    elif gender == "Prefer not to say":
      f.write("- I " + gender + " my gender. \n")
    f.write("- I live in " + live_in + "\n")

    # Personal Relationships
    f.write("Relationships of the user:" + "\n")
    f.write("- I am " + marital_stat + "\n")
    if living_situ != "Other":
        f.write("- I live " + living_situ + "\n")
    f.write("- Information about my immediate family: " + imm_family + "\n")
    if have_pets == "Yes":
        f.write("- I have/had pets." + pets_info + "\n")
    else:
        f.write("- I have not had pets. \n")
    f.write(friends_info + "\n")

    # Education and Work
    f.write("Education Qualifications of the user:" + "\n")
    f.write("- My highest level of education is " + ed_level + "\n")
    if job_y is not None and not pd.isna(job_y):
        f.write("- My profession and/or area of expertise is " + job_y + "\n")
    if job_n is not None and not pd.isna(job_n):
        f.write("- I did not do any jobs, but " + job_n + "\n")

    # Preferences
    f.write("Preferences of the user:" + "\n")
    f.write("- This is how a typical day looks like for me - " + typ_day + "\n")
    f.write("- A few activities I would like to do with my family and friends - " + act_others + "\n")
    f.write("- A few activities I would like to do with myself - " + own_act + "\n")
    f.write("- A few activities I would not like to do - " + own_act_dislike + "\n")

    # Interaction Style
    f.write("Interaction Style of the user: \n")
    f.write("- When I meet a new person for the first time, I would like to talk about - " + first_talk + "\n")
    f.write("- Interaction style I would prefer the most when I am having a conversation is " + int_style + "\n")

    # Environment and Lifestyle
    f.write("Environment & Lifestyle of the user: \n")
    f.write("- I use technology " + tech_exp + "\n")
    f.write("- Technological devices I use frequently in my life are " + tech_devices + "\n")
    if va_use_y:
        f.write("- I have used voice assistants such as " + va_type_y + "\n")
        f.write("- " + goals_y + "\n")
    else:
        f.write("I have never used voice assistants. \n")
    f.write("\n")

    # Goals for the AI by the user




