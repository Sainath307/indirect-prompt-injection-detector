import random
import pandas as pd
from mock_tools import web_search, read_file, inject_attack, INJECTION_TEMPLATES

CLEAN_TEXTS = [
    web_search("pizza recipe"),
    web_search("best laptop 2026"),
    web_search("how to train a dog"),
    web_search("weather forecast this week"),
    web_search("stock market news today"),
    read_file("notes.txt"),
    read_file("report.txt"),
    "The quarterly earnings call is scheduled for next Tuesday at 10 AM.",
    "To reset your router, hold the reset button for 10 seconds until the lights blink.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "The train departs from platform 4 at 6:45 PM and arrives by 9:15 PM.",
    "Our return policy allows exchanges within 30 days of purchase with a valid receipt.",
    "The museum is open Tuesday through Sunday, 9 AM to 5 PM, closed on Mondays.",
    "Mix two cups of flour with one teaspoon of baking soda and a pinch of salt.",
    "The new firmware update improves battery life by approximately 15 percent.",
    "To install the software, follow the on-screen setup instructions and restart your computer.",
    "The operating system update includes security patches and performance improvements.",
    "Please contact the admin office for help with your student ID card.",
    "The company's new HR system streamlines the process of submitting leave requests.",
    "Our customer support instructions are available in the help center under FAQs.",
    "The solar system consists of eight planets orbiting the sun.",
    "System requirements for this game include at least 8GB of RAM and a dedicated GPU.",
    "The admin panel lets teachers upload grades and attendance records for their class.",
    "Follow these three simple steps to set up your new smart home system.",
    "The library's cataloging system was recently upgraded to support digital lending.",
    "Our onboarding instructions walk new employees through their first week of tasks.",
    "The immune system protects the body from infections and foreign invaders.",
    "For password resets, please follow the instructions emailed to your registered address.",
    "The city's public transit system added two new bus routes this month.",
    "This recipe serves four people and takes about 45 minutes to prepare.",
    "The conference schedule includes keynote talks, workshops, and networking sessions.",
    "Battery life on the new smartwatch lasts up to five days on a single charge.",
    "The annual report shows steady growth across all three business divisions.",
    "Local weather conditions are expected to remain mild through the weekend.",
    "The hiking trail is approximately six miles long with moderate elevation gain.",
    "The recipe calls for fresh basil, garlic, and a squeeze of lemon juice.",
    "Our booking system allows you to reserve a table up to 30 days in advance.",
    "The car's onboard navigation system recalculates routes based on live traffic.",
    "New employees receive a laptop preloaded with the company's standard software.",
    "The school's grading system uses a weighted average across assignments and exams.",
    "To brew the perfect cup of coffee, use water just off the boil, around 200F.",
    "The satellite system provides GPS coverage across most of the globe.",
    "Customer instructions for assembly are included in the box along with all hardware.",
    "The banking app's security system requires two-factor authentication at login.",
    "Our irrigation system waters the garden automatically every morning at sunrise.",
    "The university's course registration system opens for new students in July.",
    "Please review the safety instructions before operating this equipment.",
    "The new operating system version rolls out gradually to all users this month.",
]

def generate_dataset(num_clean=1000, num_poisoned=1000) -> pd.DataFrame:
    rows = []
    for _ in range(num_clean):
        text = random.choice(CLEAN_TEXTS)
        rows.append({"text": text, "label": 0})

    styles = list(INJECTION_TEMPLATES.keys())
    for i in range(num_poisoned):
        base_text = random.choice(CLEAN_TEXTS)
        style = styles[i % len(styles)]
        poisoned = inject_attack(base_text, style=style)
        rows.append({"text": poisoned, "label": 1})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_dataset(num_clean=1000, num_poisoned=1000)
    df.to_csv("dataset.csv", index=False)
    print(f"Generated dataset with {len(df)} rows.")
    print(df["label"].value_counts())
