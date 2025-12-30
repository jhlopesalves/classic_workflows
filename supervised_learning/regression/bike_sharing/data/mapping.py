seasons = {
    1: "Winter",
    2: "Spring",
    3: "Summer",
    4: "Autumn",
}

years = {
    0: 2011,
    1: 2012,
}

holiday = {
    0: "holiday",
    1: "not-holiday",
}

working_day = {
    0: "normal day",
    1: "special day",
}

weather = {
    1: "Clear, Few clouds, Partly cloudy",
    2: "Mist + Cloudy, Mist + Broken clouds",
    3: "Light Snow/Rain + Thunderstorm",
    4: "Heavy Rain/Snow + Fog",
}


months = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

week_day = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}

hour = {
    0: "night",
    1: "night",
    2: "night",
    3: "night",
    4: "night",
    5: "morning",
    6: "morning",
    7: "morning",
    8: "morning",
    9: "daytime",
    10: "daytime",
    11: "daytime",
    12: "daytime",
    13: "daytime",
    14: "daytime",
    15: "daytime",
    16: "afternoon",
    17: "afternoon",
    18: "afternoon",
    19: "evening",
    20: "evening",
    21: "evening",
    22: "night",
    23: "night",
}

mapping = {
    "season": seasons,
    "yr": years,
    "holiday": holiday,
    "workingday": working_day,
    "weathersit": weather,
    "mnth": months,
    "weekday": week_day,
    "hr": hour,
}

if __name__ == "__main__":
    print("Bike Sharing Data Mappings:")
    for key, value in mapping.items():
        print(f"{key}: {value}")
