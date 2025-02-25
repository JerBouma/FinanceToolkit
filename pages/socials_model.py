"""Socials Model"""

import streamlit as st


def create_socials_sidebar():
    """
    Creates a sidebar with social media links and additional information.

    This function adds a divider and displays social media badges with links to GitHub, LinkedIn,
    GitHub Sponsors, and Buy Me a Coffee. It also includes additional information about the Finance Toolkit
    and provides a link to open a ticket for sharing thoughts or issues.

    Returns:
        Divided and Markdown objects from Streamlit.
    """
    st.divider()
    st.markdown(
        unsafe_allow_html=True,
        body="""
    [![GitHub](https://img.shields.io/badge/GitHub-grey?logo=github)](https://github.com/JerBouma/FinanceToolkit)
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-grey?logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/boumajeroen/)
    [![GitHub Sponsors](https://img.shields.io/badge/Sponsor_this_Project-grey?logo=github)](https://github.com/sponsors/JerBouma)
    [![Buy Me a Coffee](https://img.shields.io/badge/Buy_Me_a_Coffee-grey?logo=buymeacoffee)](https://www.buymeacoffee.com/jerbouma)

    \n\n The Finance Toolkit is written and maintained by [Jeroen Bouma](https://www.jeroenbouma.com).
    Like to share your thoughts? Please open a ticket [here](https://github.com/JerBouma/FinanceToolkit/issues/new/choose).
    """,
    )
