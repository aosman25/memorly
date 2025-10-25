#!/usr/bin/env python3
"""
Script to generate realistic random journals based on persons dataset.

This script:
1. Loads the persons dataset
2. Generates realistic journal entries that mention persons
3. Creates varied scenarios (activities, emotions, events)
4. Saves journals with UUID filenames in data/journals/
"""

import json
import random
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Journal templates and content pools
JOURNAL_TEMPLATES = [
    # Family gatherings
    "Had a wonderful time at the family gathering today. {person1} brought their famous {food}, and we all enjoyed it. {person2} told some hilarious stories about {topic}. It reminded me why family is so important.",

    # Daily activities
    "Went {activity} with {person1} this {time_of_day}. The weather was {weather} but we made the most of it. {person1} has such a {positive_trait}, it's always refreshing to spend time together.",

    # Celebrations
    "Today was {person1}'s {celebration}! We all gathered to celebrate. {person2} organized everything perfectly. There was {food}, music, and lots of laughter. I'm grateful for these moments.",

    # Quiet moments
    "Had a quiet {time_of_day} with {person1}. We talked about {topic} and I realized how much wisdom they have. Sometimes the simplest conversations are the most meaningful.",

    # Adventures
    "{person1} and I decided to {adventure} today. It was spontaneous and exactly what I needed. {person2} joined us later and we ended up {activity_past}. Days like these are precious.",

    # Memories
    "I was looking through old photos today and found one of {person1} from {time_past}. It made me smile thinking about all we've been through together. Called them and we reminisced about {topic}.",

    # Challenges
    "Today was challenging. {person1} helped me get through it with their {positive_trait}. I don't know what I'd do without their support. We {activity_past} afterward to decompress.",

    # Achievements
    "{person1} shared some exciting news today - they {achievement}! I'm so proud of them. We celebrated by {celebration_activity}. Their {positive_trait} really paid off.",

    # Regular days
    "A regular {day_of_week}, but special in its own way. {person1} stopped by unexpectedly. We had {food} and caught up on life. Sometimes the unplanned moments are the best.",

    # Reflections
    "Thinking about how lucky I am to have {person1} in my life. Their {positive_trait} inspires me to be better. We've shared so many {emotion} moments together.",

    # Holidays
    "The {holiday} celebration was beautiful. {person1} and {person2} were both there, along with the whole family. We {activity_past}, exchanged gifts, and made new memories.",

    # Helping each other
    "Helped {person1} with {task} today. It felt good to be useful. They've helped me so many times, it's nice to return the favor. We make a good team.",

    # Outings
    "Explored {place} with {person1} today. It was {person1}'s idea and I'm glad we went. We discovered {discovery} and spent hours just {activity_ing}. Need to do this more often.",

    # Heart-to-heart
    "Had a deep conversation with {person1} about {serious_topic}. It's rare to find someone you can be so honest with. Their perspective on {topic} really made me think.",

    # Simple pleasures
    "Shared a simple meal with {person1} - just {food} and good conversation. No phones, no distractions. Just being present. These moments ground me.",
]

# Content pools
FOODS = [
    "lasagna", "apple pie", "chocolate cake", "homemade bread", "cookies",
    "potato salad", "grilled chicken", "pasta", "sandwiches", "soup",
    "pizza", "tacos", "curry", "stir-fry", "casserole"
]

TOPICS = [
    "childhood memories", "travel adventures", "work", "hobbies",
    "books we've read", "movies", "their garden", "the old neighborhood",
    "sports", "current events", "music", "their pets", "cooking",
    "life goals", "dreams"
]

ACTIVITIES = [
    "for a walk", "hiking", "shopping", "to the park", "to the beach",
    "to a museum", "to get coffee", "for a drive", "to the gym",
    "to lunch", "to a concert", "to the library", "cycling"
]

ACTIVITIES_PAST = [
    "watching the sunset", "playing board games", "cooking together",
    "going for ice cream", "taking photos", "listening to music",
    "working on a project", "gardening", "organizing old photos",
    "baking cookies"
]

ACTIVITIES_ING = [
    "talking", "walking around", "exploring", "taking pictures",
    "enjoying the atmosphere", "people watching", "relaxing"
]

ADVENTURES = [
    "visit that new cafe downtown", "explore the hiking trail",
    "check out the farmers market", "go to the antique shop",
    "visit the art gallery", "try that new restaurant",
    "go to the bookstore"
]

WEATHER = [
    "perfect", "a bit chilly", "sunny", "cloudy", "rainy",
    "unexpectedly nice", "typical for this time of year"
]

POSITIVE_TRAITS = [
    "positive attitude", "sense of humor", "kindness", "wisdom",
    "energy", "patience", "generosity", "warmth", "honesty",
    "enthusiasm", "creativity", "reliability", "compassion"
]

CELEBRATIONS = [
    "birthday", "promotion", "anniversary", "graduation",
    "retirement party", "housewarming"
]

CELEBRATION_ACTIVITIES = [
    "going out to dinner", "having cake and coffee",
    "raising a glass of champagne", "throwing a small party",
    "going to their favorite place"
]

ACHIEVEMENTS = [
    "got promoted", "finished their degree", "started a new business",
    "completed a marathon", "published an article", "learned a new skill",
    "bought a house", "got a new job"
]

TASKS = [
    "moving furniture", "fixing their computer", "painting a room",
    "organizing the garage", "planning an event", "car repairs",
    "gardening", "sorting through old boxes"
]

PLACES = [
    "the botanical garden", "downtown", "the old neighborhood",
    "the riverfront", "the historic district", "the farmer's market",
    "a nearby town", "the state park"
]

DISCOVERIES = [
    "a great coffee shop", "a beautiful view", "an interesting store",
    "a quiet spot", "some local art", "a hidden path"
]

SERIOUS_TOPICS = [
    "life changes", "future plans", "family matters", "career decisions",
    "health", "relationships", "personal growth", "challenges we're facing"
]

EMOTIONS = [
    "joyful", "challenging", "meaningful", "memorable", "heartwarming",
    "difficult", "beautiful", "precious"
]

TIMES_OF_DAY = [
    "morning", "afternoon", "evening"
]

DAYS_OF_WEEK = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]

TIMES_PAST = [
    "years ago", "last summer", "when we were younger", "a long time ago",
    "back in the day"
]

HOLIDAYS = [
    "Christmas", "Thanksgiving", "New Year", "Easter", "Fourth of July",
    "birthday"
]


def load_persons_dataset(dataset_path: Path) -> Dict:
    """Load the persons dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def get_person_reference(person_data: Dict) -> str:
    """Get a reference string for a person (name or relationship)."""
    if person_data['name']:
        return person_data['name']
    elif person_data['relationship']:
        return f"my {person_data['relationship'].lower()}"
    else:
        return "my friend"


def generate_journal_entry(persons_list: List[Dict]) -> str:
    """Generate a realistic journal entry using persons from the dataset."""
    # Choose a random template
    template = random.choice(JOURNAL_TEMPLATES)

    # Select persons to mention (1-2 persons)
    num_persons = min(random.randint(1, 2), len(persons_list))
    selected_persons = random.sample(persons_list, num_persons)

    # Fill in the template
    replacements = {
        'person1': get_person_reference(selected_persons[0]),
        'person2': get_person_reference(selected_persons[1]) if num_persons > 1 else get_person_reference(selected_persons[0]),
        'food': random.choice(FOODS),
        'topic': random.choice(TOPICS),
        'activity': random.choice(ACTIVITIES),
        'activity_past': random.choice(ACTIVITIES_PAST),
        'activity_ing': random.choice(ACTIVITIES_ING),
        'adventure': random.choice(ADVENTURES),
        'weather': random.choice(WEATHER),
        'positive_trait': random.choice(POSITIVE_TRAITS),
        'celebration': random.choice(CELEBRATIONS),
        'celebration_activity': random.choice(CELEBRATION_ACTIVITIES),
        'achievement': random.choice(ACHIEVEMENTS),
        'task': random.choice(TASKS),
        'place': random.choice(PLACES),
        'discovery': random.choice(DISCOVERIES),
        'serious_topic': random.choice(SERIOUS_TOPICS),
        'emotion': random.choice(EMOTIONS),
        'time_of_day': random.choice(TIMES_OF_DAY),
        'day_of_week': random.choice(DAYS_OF_WEEK),
        'time_past': random.choice(TIMES_PAST),
        'holiday': random.choice(HOLIDAYS),
    }

    entry = template.format(**replacements)
    return entry


def generate_timestamp() -> int:
    """Generate a random Unix timestamp within the past year."""
    days_ago = random.randint(0, 365)
    date = datetime.now() - timedelta(days=days_ago)
    return int(date.timestamp())


def generate_journals(persons_dataset: Dict, num_journals: int, output_dir: Path):
    """Generate random journal entries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert persons to list (only include those with names or relationships)
    persons_list = [
        person for person in persons_dataset.values()
        if person['name'] or person['relationship']
    ]

    if not persons_list:
        print("No named persons found in dataset. Cannot generate journals.")
        return

    print(f"Generating {num_journals} journal entries...")
    print(f"Using {len(persons_list)} persons from dataset\n")

    for i in range(num_journals):
        # Generate journal entry
        entry = generate_journal_entry(persons_list)
        timestamp = generate_timestamp()

        # Create journal object
        journal_id = str(uuid.uuid4())
        journal = {
            "id": journal_id,
            "timestamp": timestamp,
            "entry": entry
        }

        # Save to file
        output_path = output_dir / f"{journal_id}.json"
        with open(output_path, 'w') as f:
            json.dump(journal, f, indent=2)

        # Convert timestamp to readable date for display
        readable_date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{i+1}/{num_journals}] Generated: {journal_id}.json")
        print(f"  Timestamp: {timestamp} ({readable_date})")
        print(f"  Entry: {entry[:80]}..." if len(entry) > 80 else f"  Entry: {entry}")
        print()

    print(f"\n✓ Successfully generated {num_journals} journals in {output_dir}/")


def main():
    """Main entry point."""
    print("Journal Generator")
    print("=" * 60)

    # Set paths
    script_dir = Path(__file__).parent  # data/scripts/
    persons_dataset_path = script_dir.parent / "persons" / "dataset.json"
    journals_dir = script_dir.parent / "journals"

    if not persons_dataset_path.exists():
        print(f"Error: Persons dataset not found at {persons_dataset_path}")
        return

    # Load persons dataset
    persons_dataset = load_persons_dataset(persons_dataset_path)
    print(f"Loaded {len(persons_dataset)} persons from dataset\n")

    # Generate journals (default: 20 journals, can be changed)
    num_journals = 20
    generate_journals(persons_dataset, num_journals, journals_dir)


if __name__ == "__main__":
    main()
