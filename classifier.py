def classify_article(title, description):
    text = (title + " " + description).lower()

    if "sport" in text:
        return "Sports"
    elif "politic" in text:
        return "Politics"
    elif "tech" in text or "ai" in text:
        return "Technology"
    else:
        return "Unknown"


def extract_tags(title, description):
    text = (title + " " + description).lower()
    tags = []

    keywords = ["ai", "python", "football", "economy", "war", "crypto"]

    for k in keywords:
        if k in text:
            tags.append(k)

    return tags