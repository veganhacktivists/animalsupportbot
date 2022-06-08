"""
This file contains the templates for responses used in redditbot.py
"""

# End of the response template
## Contains 1 formattable field (for link references)
END_TEMPLATE = """
Learn more at **[WatchDominion](https://watchdominion.org/), [Vegan Bootcamp](https://vbcamp.org/04eed1)**
\n *** \n This was an automatically generated response based on the idea(s)/myth(s): \n\n {} \n\n
\n *** \n
*If you would like to contribute to the bot, please take a look at the GitHub repo: [veganhacktivists/animalsupportbot](https://github.com/veganhacktivists/animalsupportbot)* \n\n
"""

# Hint info template
## Contains 1 formattable field
HINT_TEMPLATE = """
The bot was hinted towards the following argument(s): {}
"""

# Respond to mention with this comment when bot fails
## No forrmattable fields
FAILURE_COMMENT = """
We're sorry! We scanned the post you replied to but didn't find any matching phrases that we could respond to.

If you think it should have responded differently, raise this as an issue on the [GitHub repo](https://github.com/veganhacktivists/animalsupportbot/issues).
"""

# Failure message to PM user
## Contains 1 formattable field (for failed comment)
FAILURE_PM = """
Hi, we couldn't find a response to the following comment: \n
"{}" \n
If you think this response should have been responded to automatically, please report this as an issue on the [GitHub repo](https://github.com/veganhacktivists/animalsupportbot/issues)
"""
