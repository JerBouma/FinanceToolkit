# CONTRIBUTING
First off all, thank you for taking the time to contribute (or at least read the Contributing Guidelines)! 🚀

The goal of the Finance Toolkit is to make any type of financial calculation as transparent and efficient as possible. I want to make these type of calculations as accessible to anyone as possible and seeing how many websites exists that do the same thing (but instead you have to pay) gave me plenty of reasons to work on this.

The following is a set of guidelines for contributing to the Finance Toolkit.

## Working with Git & Pull Requests

Any new contribution preferably goes via a Pull Request. In essence, all you really need is Git and basic understanding of how a Pull Request works. Find some resources that explain this well here:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

On every Pull Request, a couple of linters will run (see [here](https://github.com/JerBouma/FinanceDatabase/blob/main/.github/workflows/) as well as categorization and compression linters). The linters check the code and whether it matches specific coding formatting. This is entirely irrelevant for the database itself but keeps the code of the related package in check as well as any markdown changes. The categorization and compression actions are very relevant for the database as it makes it much easier and faster to read data.

## Following the Workflow

After setting up Git, you can fork and pull the project in.

1. Fork the Project ([more info](https://docs.github.com/en/get-started/quickstart/fork-a-repo))
    - **Using GitHub Desktop:** [Getting started with GitHub Desktop](https://docs.github.com/en/desktop/installing-and-configuring-github-desktop/getting-started-with-github-desktop) will guide you through setting up Desktop. Once Desktop is set up, you can use it to [fork the repo](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/cloning-and-forking-repositories-from-github-desktop)!
    - **Using the command line:** [Fork the repo](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo#fork-an-example-repository) so that you can make your changes without affecting the original project until you're ready to merge them.
2. Pull the Repository Locally ([more info](https://github.com/git-guides/git-pull))
3. Create your own branch (`git checkout -b feature/contribution`)
4. Add your changes (`git add .`)
5. Install pre-commit, this checks the code for any errors before committing (`pre-commit install`)
6. Commit your Changes (`git commit -m 'Improve the Toolkit'`)
7. Check whether the tests still pass (`pytest tests`)`
5. Push to your Branch (`git push origin feature/contribution`)
6. Open a Pull Request

Note: feel free to reach out if you run into any issues: jer.bouma@gmail.com or https://www.linkedin.com/in/boumajeroen/ or open a GitHub Issue.