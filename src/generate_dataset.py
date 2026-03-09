"""
generate_dataset.py
-------------------
Generates a realistic synthetic support ticket dataset.
In production, replace this with your Kaggle dataset (e.g., suraj520/customer-support-ticket-dataset).
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ── Ticket templates per category ──────────────────────────────────────────

TEMPLATES = {
    "Billing": {
        "High": [
            "I was charged twice for my subscription this month. Please refund immediately.",
            "My credit card was billed {amount} without authorization. This needs urgent attention.",
            "Double billing occurred on my account. Invoice #{inv}. Need immediate resolution.",
            "Unauthorized charge of {amount} on my account. I demand a refund now.",
            "My payment failed but I was still charged {amount}. This is urgent.",
        ],
        "Medium": [
            "I have a question about my invoice #{inv} from last month.",
            "Can you explain the {amount} charge on my latest billing statement?",
            "I need to update my billing information and payment method.",
            "My subscription renewal date changed without notice. Please clarify.",
            "I would like to downgrade my plan and understand the billing impact.",
            "Please send me an itemized invoice for my account #{inv}.",
        ],
        "Low": [
            "When does my billing cycle reset each month?",
            "Do you offer annual billing discounts for enterprise plans?",
            "What payment methods do you accept?",
            "Can I get a receipt for my last payment?",
            "How do I add a secondary billing contact to my account?",
        ],
    },
    "Technical Issue": {
        "High": [
            "Your service is completely down. We cannot access the platform at all. This is a production outage.",
            "Critical bug: all our user data appears corrupted after the last update. URGENT.",
            "API returning 500 errors for all requests since {time}. Our entire workflow is blocked.",
            "Login is broken for all users in our organization. Nobody can access the system.",
            "Data loss detected after the recent migration. Many records are missing.",
        ],
        "Medium": [
            "The export feature is not working correctly. Files download but are empty.",
            "I keep getting a 404 error when trying to access the reports section.",
            "The mobile app crashes whenever I try to upload an image.",
            "Dashboard widgets are not loading properly on Chrome browser.",
            "Integration with Slack stopped working after the recent update.",
            "Search functionality returns incorrect results for certain queries.",
        ],
        "Low": [
            "The UI looks slightly broken on Safari — minor layout issue.",
            "Notification emails are arriving with some formatting issues.",
            "Dark mode has a small color inconsistency on the settings page.",
            "The tooltip text on the help icon appears to be truncated.",
            "Minor lag when switching between tabs in the dashboard.",
        ],
    },
    "Account": {
        "High": [
            "My account has been locked and I cannot access critical business data. Please unlock immediately.",
            "I suspect my account has been compromised. There are logins from unknown locations.",
            "All admin users have been accidentally removed. We are locked out of our organization.",
            "Account deletion was triggered by mistake. Please restore our data immediately.",
            "Two-factor authentication is broken and we cannot log in at all.",
        ],
        "Medium": [
            "I need to transfer my account to a new email address.",
            "How do I add a team member with limited permissions to my account?",
            "I need to update my company details and billing address on the account.",
            "Please help me merge two accounts that were created accidentally.",
            "I forgot my password and the reset email is not arriving.",
            "How do I enable SSO (Single Sign-On) for my organization?",
        ],
        "Low": [
            "How do I change my display name on my profile?",
            "Can I add a profile picture to my account?",
            "How do I update my notification preferences?",
            "Where can I find my account ID?",
            "How do I switch between different workspaces in my account?",
        ],
    },
    "General Query": {
        "High": [
            "We are evaluating your platform for 500+ users. Need urgent response for a board decision tomorrow.",
            "Compliance audit requires documentation of your data processing by end of today.",
            "Legal team needs your DPA (Data Processing Agreement) urgently for contract signing.",
        ],
        "Medium": [
            "Can you explain how your data backup and recovery process works?",
            "What is your uptime SLA and how do you handle planned maintenance?",
            "Does your platform support GDPR compliance features?",
            "What integrations do you support with third-party tools?",
            "Can you provide a comparison between your Business and Enterprise plans?",
            "How does your pricing scale for larger teams?",
        ],
        "Low": [
            "Do you have a public API documentation page?",
            "What are your customer support hours?",
            "Do you offer a free trial for new users?",
            "Is there a mobile app available for iOS and Android?",
            "Where can I find tutorial videos for getting started?",
            "Do you have a community forum or help center?",
            "What languages does your platform support?",
        ],
    },
    "Feature Request": {
        "High": [
            "Our entire team is blocked without bulk export functionality. This is business-critical.",
            "We need audit logs for compliance — without this we cannot use your service.",
        ],
        "Medium": [
            "It would be great to have a calendar view for task management.",
            "Please add the ability to export reports in Excel format.",
            "We would love a dark mode option for the web interface.",
            "Can you add keyboard shortcuts for common actions in the editor?",
            "Please consider adding custom fields to ticket forms.",
            "A recurring task feature would significantly improve our workflow.",
        ],
        "Low": [
            "Would be nice to have more color themes available.",
            "Could you add drag-and-drop support for file attachments?",
            "Please add the ability to pin important messages in channels.",
            "It would be helpful to have a 'recently viewed' section on the dashboard.",
            "Could you support emoji reactions on comments?",
        ],
    },
}

FILLERS = {
    "amount": ["$49.99", "$99.00", "$199.00", "$29.95", "$149.00", "$299.00"],
    "inv": ["INV-2024-0{n}".format(n=str(random.randint(100, 999))) for _ in range(20)],
    "time": ["2:00 AM UTC", "this morning", "6 hours ago", "yesterday evening", "10:30 PM"],
}


def fill_template(text):
    for key, options in FILLERS.items():
        if "{" + key + "}" in text:
            text = text.replace("{" + key + "}", random.choice(options))
    return text


def generate_tickets(n=2000):
    rows = []
    categories = list(TEMPLATES.keys())
    priorities = ["High", "Medium", "Low"]

    # Weighted distribution to match realistic imbalance
    cat_weights = [0.25, 0.30, 0.20, 0.15, 0.10]
    pri_weights = [0.25, 0.45, 0.30]

    for i in range(n):
        category = random.choices(categories, weights=cat_weights)[0]
        priority = random.choices(priorities, weights=pri_weights)[0]

        templates_available = TEMPLATES[category].get(priority, TEMPLATES[category]["Low"])
        text = fill_template(random.choice(templates_available))

        # Add realistic noise / variation
        prefixes = [
            "Hi support team, ",
            "Hello, ",
            "Good morning, ",
            "To whom it may concern, ",
            "Urgent: ",
            "Hello there! ",
            "",
            "",
            "",
        ]
        suffixes = [
            " Please help.",
            " Thanks in advance.",
            " Looking forward to your response.",
            " This is affecting my work.",
            " I need this resolved ASAP.",
            "",
            "",
            "",
        ]

        text = random.choice(prefixes) + text + random.choice(suffixes)

        rows.append({
            "ticket_id": f"TKT-{10000 + i}",
            "text": text.strip(),
            "category": category,
            "priority": priority,
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_tickets(2000)
    df.to_csv("data/support_tickets.csv", index=False)
    print(f"✅ Dataset generated: {len(df)} tickets")
    print(df["category"].value_counts())
    print(df["priority"].value_counts())
