"""
predict.py
-----------
Inference module: classify and prioritize new support tickets.
Load trained models and predict on any new text input.
"""

import os
import sys
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from text_preprocessor import clean_text


class TicketClassifier:
    """
    Production-ready classifier for support tickets.

    Usage:
        clf = TicketClassifier()
        result = clf.predict("I was charged twice this month!")
        print(result)
        # → {"category": "Billing", "priority": "High", "confidence": 0.94}
    """

    PRIORITY_EMOJI = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    CATEGORY_EMOJI = {
        "Billing": "💳",
        "Technical Issue": "🔧",
        "Account": "👤",
        "General Query": "❓",
        "Feature Request": "💡",
    }

    def __init__(self):
        self.tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        self.cat_model = joblib.load(os.path.join(MODEL_DIR, "category_classifier.pkl"))
        self.pri_model = joblib.load(os.path.join(MODEL_DIR, "priority_classifier.pkl"))
        print("✅ Models loaded successfully.")

    def predict(self, raw_text: str) -> dict:
        """Predict category and priority for a single ticket."""
        cleaned = clean_text(raw_text)
        if not cleaned.strip():
            return {"error": "Ticket text is empty after cleaning."}

        vec = self.tfidf.transform([cleaned])

        # Category
        cat_pred = self.cat_model.predict(vec)[0]
        cat_proba = self.cat_model.predict_proba(vec)[0]
        cat_confidence = max(cat_proba)
        cat_classes = self.cat_model.classes_

        # Priority
        pri_pred = self.pri_model.predict(vec)[0]
        pri_proba = self.pri_model.predict_proba(vec)[0]
        pri_confidence = max(pri_proba)
        pri_classes = self.pri_model.classes_

        return {
            "raw_text": raw_text[:120] + ("..." if len(raw_text) > 120 else ""),
            "clean_text": cleaned,
            "category": cat_pred,
            "category_confidence": round(float(cat_confidence), 3),
            "category_scores": {
                cls: round(float(p), 3)
                for cls, p in zip(cat_classes, cat_proba)
            },
            "priority": pri_pred,
            "priority_confidence": round(float(pri_confidence), 3),
            "priority_scores": {
                cls: round(float(p), 3)
                for cls, p in zip(pri_classes, pri_proba)
            },
        }

    def predict_batch(self, texts: list) -> list:
        """Predict for a list of ticket texts."""
        return [self.predict(t) for t in texts]

    def display(self, result: dict):
        """Pretty-print a prediction result."""
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        cat = result["category"]
        pri = result["priority"]
        cat_emoji = self.CATEGORY_EMOJI.get(cat, "📌")
        pri_emoji = self.PRIORITY_EMOJI.get(pri, "⚪")

        print(f"\n{'─'*55}")
        print(f"  🎫 TICKET: {result['raw_text']}")
        print(f"{'─'*55}")
        print(f"  {cat_emoji} Category : {cat:<20} (confidence: {result['category_confidence']:.0%})")
        print(f"  {pri_emoji} Priority : {pri:<20} (confidence: {result['priority_confidence']:.0%})")
        print(f"\n  Category breakdown:")
        for cls, score in sorted(result["category_scores"].items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            print(f"    {cls:<20} {bar:<20} {score:.0%}")
        print(f"\n  Priority breakdown:")
        for cls, score in sorted(result["priority_scores"].items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            print(f"    {cls:<20} {bar:<20} {score:.0%}")
        print(f"{'─'*55}")


# ── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    clf = TicketClassifier()

    test_tickets = [
        "URGENT: Our entire production server is down. No users can log in. We are losing revenue every minute!",
        "Hi, I noticed an extra $99 charge on my account this month that I didn't authorize. Can you look into this?",
        "How do I change my account password? I forgot it.",
        "It would be great if you could add a dark mode to the dashboard.",
        "We need the data export compliance report for our ISO audit by end of day.",
        "The chart on the analytics page looks a bit off in Firefox, but it works fine in Chrome.",
    ]

    print("\n" + "=" * 60)
    print("  SUPPORT TICKET CLASSIFICATION DEMO")
    print("=" * 60)

    for ticket in test_tickets:
        result = clf.predict(ticket)
        clf.display(result)
