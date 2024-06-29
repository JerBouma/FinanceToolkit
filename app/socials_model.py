"""Socials Model"""

import streamlit as st


def create_socials_sidebar():
    st.divider()
    st.markdown(
        unsafe_allow_html=True,
        body="""
    [![GitHub](https://img.shields.io/badge/GitHub-grey?logo=github)](https://github.com/JerBouma/FinanceToolkit)
    [![Twitter](https://img.shields.io/badge/Twitter-grey?logo=x)](https://twitter.com/JerBouma)
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-grey?logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/boumajeroen/)
    [![GitHub Sponsors](https://img.shields.io/badge/Sponsor_this_Project-grey?logo=github)](https://github.com/sponsors/JerBouma)
    [![Buy Me a Coffee](https://img.shields.io/badge/Buy_Me_a_Coffee-grey?logo=buymeacoffee)](https://www.buymeacoffee.com/jerbouma)

    \n\n The Finance Toolkit is written and maintained by [Jeroen Bouma](https://www.jeroenbouma.com)
    """,
    )
