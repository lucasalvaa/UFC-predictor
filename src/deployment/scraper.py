import re
from datetime import datetime
from typing import Dict

import numpy as np
import requests
from bs4 import BeautifulSoup


def get_fighter_url(full_name: str) -> str | None:
    """Search for the fighter on ufcstats.com/statistics/fighters
    and, if it finds a match, returns the URL of the fighter's page.

    :param full_name: The full name of the fighter.
        Although not recommended, it can also accept only the first name or last name.
    :return: The URL of the fighter's page.
    """
    # Split name and last name
    parts = full_name.strip().split()
    if len(parts) < 2:
        last_name_query = parts[0]
        first_name_query = ""
    else:
        last_name_query = parts[-1]
        first_name_query = parts[0]

    # Send request to ufcstats using the righter part of the full name
    base_url = "http://ufcstats.com/statistics/fighters/search"
    params = {"query": last_name_query}

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Connection error: {e}")
        return None

    soup = BeautifulSoup(resp.content, "html.parser")
    rows = soup.find_all("tr", class_="b-statistics__table-row")

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 2:
            continue

        link_tag = cols[0].find("a")
        if not link_tag:
            continue

        table_first_name = cols[0].get_text(strip=True).lower()
        table_last_name = cols[1].get_text(strip=True).lower()

        # Check if the fighter's name corresponds
        if (
            first_name_query.lower() in table_first_name
            and last_name_query.lower() in table_last_name
        ):
            return link_tag["href"]  # Link Found

    return None


def extract_fight_history(soup: BeautifulSoup) -> Dict[str, int]:
    """Calculate a fighter's wins and losses by KO and submission
    from the fighter's match history table on ufcstats.com.

    :param soup: The BeautifulSoup soup of the fighter's match history table.
    :return: A dictionary whose keys are
        "wins_ko", "losses_ko", "wins_sub", "losses_sub".
    """
    # Initialize the dictionary
    stats = {"wins_ko": 0, "wins_sub": 0, "losses_ko": 0, "losses_sub": 0}

    # Get al the rows of the fighter history's table
    # The table's rows have the class 'b-fight-details__table-row'
    rows = soup.find_all("tr", class_="b-fight-details__table-row")

    for row in rows:
        cols = row.find_all("td")

        if len(cols) < 8:
            continue

        # Extract the fighter outcome in the match
        outcome_text = cols[0].get_text(strip=True).lower()

        # Extract the fight ending's method
        method_text = cols[7].get_text(strip=True).lower()

        # Update the dictionary
        if "win" == outcome_text:
            if "ko" in method_text or "tko" in method_text:
                stats["wins_ko"] += 1
            elif "sub" in method_text:
                stats["wins_sub"] += 1

        elif "loss" == outcome_text:
            if "ko" in method_text or "tko" in method_text:
                stats["losses_ko"] += 1
            elif "sub" in method_text:
                stats["losses_sub"] += 1

    return stats


def calculate_age(dob_string: str) -> float | None:
    """Convert a string formatted ad 'MMM dd, yyyy' into an age.

    :param dob_string: A string formatted as 'MMM dd, yyyy'
    :return: Age in years.
    """
    if not dob_string or dob_string == "--":
        return None

    try:
        dob_dt = datetime.strptime(dob_string, "%b %d, %Y")
        return (datetime.today() - dob_dt).days / 365.25
    except Exception as e:
        print(f"An error occurred parsing the date '{dob_string}': {e}")
        return None


def search_fighter_stats(fighter_name: str) -> Dict[str, str | float | int] | None:
    """Search fighter stats on ufcstats.com given his name.

    :param fighter_name: The full name of the fighter.
        Although not recommended, it can also accept only the first name or last name.
    :return: A dictionary whose keys are "name", "height_cm", "weight_lbs",
        "reach_cm", "age", "stance", "wins", "losses", "draws",
        "wins_ko", "losses_ko", "wins_sub", "losses_sub".
    """
    url = get_fighter_url(fighter_name)

    if not url:
        print(f"No URL found for {fighter_name}")
        return None

    # HTTP request to the fighter's page on ufcstats.com
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, "html.parser")

    # Initialize dictionary
    stats: Dict[str, str | float | int] = {"name": fighter_name}

    info_box = soup.find("div", class_="b-list__info-box")
    items = info_box.find_all("li")

    for i in items:
        text = i.text.strip().replace("\n", "")
        if "Height:" in text:
            # Height is converted in centimeters
            h_clean = text.replace("Height:", "").strip()
            try:
                ft, inch = h_clean.split("'")
                inch = inch.replace('"', "").strip()
                stats["height_cm"] = (int(ft) * 30.48) + (int(inch) * 2.54)
            except ValueError:
                stats["height_cm"] = np.nan

        elif "Reach:" in text:
            # Reach is converted in centimeters
            r_clean = text.replace("Reach:", "").replace('"', "").strip()
            if r_clean != "--" and r_clean != "":
                stats["reach_cm"] = float(r_clean) * 2.54
            else:
                # If reach is missing, height is often used instead
                stats["reach_cm"] = stats.get("height_cm", np.nan)

        elif "Weight:" in text:
            w_clean = text.replace("Weight:", "").replace("lbs.", "").strip()

            try:
                if w_clean and w_clean != "--":
                    stats["weight_lbs"] = float(w_clean)
                else:
                    stats["weight_lbs"] = np.nan
            except ValueError:
                stats["weight_lbs"] = np.nan

        elif "STANCE:" in text:
            stats["stance"] = text.replace("STANCE:", "").strip()

        elif "DOB:" in text:
            dob_raw = text.replace("DOB:", "").strip()
            stats["age"] = calculate_age(dob_raw)

    # Parsing the record (W-L-D)
    record_text = soup.find("span", class_="b-content__title-record").text.strip()
    record_match = re.search(r"Record:\s*(\d+)-(\d+)-(\d+)", record_text)
    if record_match:
        stats["wins"] = int(record_match.group(1))
        stats["losses"] = int(record_match.group(2))
        stats["draws"] = int(record_match.group(3))

    # Add historical wins and losses by KO and Sumbission to the dictionary
    stats.update(extract_fight_history(soup))

    return stats
