import numpy as np
from pandas.api.types import CategoricalDtype


class ColumnDescription:
    def __init__(self, full_name, name, dtype=None, converter=None, usecol=True):
        self.full_name = full_name
        self.name = name
        self.dtype = dtype
        self.converter = converter
        self.usecol = usecol


def notNA(text):
    return text != 'NA'


CATEGORICAL_USAGE = CategoricalDtype([
    "Never or Almost never",
    "From time to time",
    "Often"
], ordered=True)


DATA_DICTIONARY = [
    ColumnDescription(
        # People who selected 'No, I don't use Python' were excluded from
        # the rest of the survey.

        full_name='Is Python the main language you use for your current projects?',
        name='python_main',
        dtype=CategoricalDtype()
    ),

    ColumnDescription(
        full_name="None:What other language(s) do you use?",
        name="otherlang_none",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Java:What other language(s) do you use?",
        name="otherlang_java",
        converter=notNA
    ),

    ColumnDescription(
        full_name="JavaScript:What other language(s) do you use?",
        name="otherlang_js",
        converter=notNA
    ),

    ColumnDescription(
        full_name="C/C++:What other language(s) do you use?",
        name="otherlang_c",
        converter=notNA
    ),

    ColumnDescription(
        full_name="PHP:What other language(s) do you use?",
        name="otherlang_php",
        converter=notNA
    ),

    ColumnDescription(
        full_name="C#:What other language(s) do you use?",
        name="otherlang_cs",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Ruby:What other language(s) do you use?",
        name="otherlang_ruby",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Bash / Shell:What other language(s) do you use?",
        name="otherlang_bash",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Objective-C:What other language(s) do you use?",
        name="otherlang_objc",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Go:What other language(s) do you use?",
        name="otherlang_go",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Visual Basic:What other language(s) do you use?",
        name="otherlang_vb",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Scala:What other language(s) do you use?",
        name="otherlang_scala",
        converter=notNA
    ),

    ColumnDescription(
        full_name="SQL:What other language(s) do you use?",
        name="otherlang_sql",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Kotlin:What other language(s) do you use?",
        name="otherlang_kotlin",
        converter=notNA
    ),

    ColumnDescription(
        full_name="R:What other language(s) do you use?",
        name="otherlang_r",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Swift:What other language(s) do you use?",
        name="otherlang_swift",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Clojure:What other language(s) do you use?",
        name="otherlang_clojure",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Perl:What other language(s) do you use?",
        name="otherlang_perl",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Rust:What other language(s) do you use?",
        name="otherlang_rust",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Groovy:What other language(s) do you use?",
        name="otherlang_groovy",
        converter=notNA
    ),

    ColumnDescription(
        full_name="TypeScript:What other language(s) do you use?",
        name="otherlang_ts",
        converter=notNA
    ),

    ColumnDescription(
        full_name="CoffeeScript:What other language(s) do you use?",
        name="otherlang_coffee",
        converter=notNA
    ),

    ColumnDescription(
        full_name="HTML/CSS:What other language(s) do you use?",
        name="otherlang_html",
        converter=notNA
    ),

    ColumnDescription(
        # This column indicates that the user wrote in that they use
        # something we didn't offer as an option. Write-in data was
        # cleaned out for data protection.

        full_name="Other - Write In::What other language(s) do you use?",
        name="otherlang_other",
        usecol=False
    ),

    ColumnDescription(
        full_name="Educational purposes: What do you use Python for?",
        name="usecase_edu",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Data analysis: What do you use Python for?",
        name="usecase_data",
        converter=notNA
    ),

    ColumnDescription(
        full_name="DevOps / System administration / Writing automation scripts: What do you use Python for?",
        name="usecase_devops",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Software testing / Writing automated tests: What do you use Python for?",
        name="usecase_testing",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Software prototyping: What do you use Python for?",
        name="usecase_prototyping",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Web development: What do you use Python for?",
        name="usecase_web",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Machine learning: What do you use Python for?",
        name="usecase_ml",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Mobile development: What do you use Python for?",
        name="usecase_mobile",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Desktop development: What do you use Python for?",
        name="usecase_desktop",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Computer graphics: What do you use Python for?",
        name="usecase_graphics",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Network programming: What do you use Python for?",
        name="usecase_network",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Game development: What do you use Python for?",
        name="usecase_game",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Multimedia applications development: What do you use Python for?",
        name="usecase_multimedia",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Embedded development: What do you use Python for?",
        name="usecase_embedded",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Programming of web parsers / scrapers / crawlers: What do you use Python for?",
        name="usecase_scraping",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Other - Write In:: What do you use Python for?",
        name="usecase_other",
        usecol=False
    ),

    ColumnDescription(
        full_name="What do you use Python for the most?",
        name="What do you use Python for the most?",
        dtype=CategoricalDtype()
    ),

    ColumnDescription(
        full_name="Which version of Python do you use the most?",
        name="Which version of Python do you use the most?",
        dtype=CategoricalDtype()
    ),

    ColumnDescription(
        full_name="Installer from python.org:What do you typically use to upgrade your Python version?",
        name="upgradepy_pythonorg",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Build from source:What do you typically use to upgrade your Python version?",
        name="upgradepy_source",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Automatic upgrade via cloud provider:What do you typically use to upgrade your Python version?",
        name="upgradepy_cloud",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Enthought:What do you typically use to upgrade your Python version?",
        name="upgradepy_enthought",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Anaconda:What do you typically use to upgrade your Python version?",
        name="upgradepy_anaconda",
        converter=notNA
    ),

    ColumnDescription(
        full_name="ActivePython:What do you typically use to upgrade your Python version?",
        name="upgradepy_activepython",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Intel Distribution for Python:What do you typically use to upgrade your Python version?",
        name="upgradepy_intel",
        converter=notNA
    ),

    ColumnDescription(
        full_name="OS-provided Python (via apt-get, yum, homebrew, etc.):What do you typically use to upgrade your Python version?",
        name="upgradepy_os",
        converter=notNA
    ),

    ColumnDescription(
        full_name="pyenv:What do you typically use to upgrade your Python version?",
        name="upgradepy_pyenv",
        converter=notNA
    ),

    ColumnDescription(
        full_name="pythonz:What do you typically use to upgrade your Python version?",
        name="upgradepy_pythonz",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Other - Write In::What do you typically use to upgrade your Python version?",
        name="upgradepy_other",
        usecol=False
    ),

    ColumnDescription(
        full_name="None:What framework(s) do you use in addition to Python?",
        name="frameworks_none",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Django:What framework(s) do you use in addition to Python?",
        name="frameworks_django",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Flask:What framework(s) do you use in addition to Python?",
        name="frameworks_flask",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Tornado:What framework(s) do you use in addition to Python?",
        name="frameworks_tornado",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Bottle:What framework(s) do you use in addition to Python?",
        name="frameworks_bottle",
        converter=notNA
    ),

    ColumnDescription(
        full_name="web2py:What framework(s) do you use in addition to Python?",
        name="frameworks_web2py",
        converter=notNA
    ),

    ColumnDescription(
        full_name="NumPy / pandas / Matplotlib / scipy and similar:What framework(s) do you use in addition to Python?",
        name="frameworks_science",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Keras / Theano / TensorFlow / scikit-learn and similar:What framework(s) do you use in addition to Python?",
        name="frameworks_ml",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Pillow:What framework(s) do you use in addition to Python?",
        name="frameworks_pillow",
        converter=notNA
    ),

    ColumnDescription(
        full_name="PyQT / PyGTK / wxPython:What framework(s) do you use in addition to Python?",
        name="frameworks_window",
        converter=notNA
    ),

    ColumnDescription(
        full_name="TkInter:What framework(s) do you use in addition to Python?",
        name="frameworks_tkinter",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Pygame:What framework(s) do you use in addition to Python?",
        name="frameworks_pygame",
        converter=notNA
    ),

    ColumnDescription(
        full_name="CherryPy:What framework(s) do you use in addition to Python?",
        name="frameworks_cherrypy",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Twisted:What framework(s) do you use in addition to Python?",
        name="frameworks_twisted",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Pyramid:What framework(s) do you use in addition to Python?",
        name="frameworks_pyramid",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Requests:What framework(s) do you use in addition to Python?",
        name="frameworks_requests",
        converter=notNA
    ),

    ColumnDescription(
        full_name="asyncio:What framework(s) do you use in addition to Python?",
        name="frameworks_asyncio",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Kivy:What framework(s) do you use in addition to Python?",
        name="frameworks_kivy",
        converter=notNA
    ),

    ColumnDescription(
        full_name="six:What framework(s) do you use in addition to Python?",
        name="frameworks_six",
        converter=notNA
    ),

    ColumnDescription(
        full_name="aiohttp:What framework(s) do you use in addition to Python?",
        name="frameworks_aiohttp",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Other - Write In::What framework(s) do you use in addition to Python?",
        name="frameworks_other",
        usecol=False
    ),

    ColumnDescription(
        full_name="None:What additional technology(s) do you use in addition to Python?",
        name="tech_none",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Jupyter Notebook:What additional technology(s) do you use in addition to Python?",
        name="tech_ipynb",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Anaconda:What additional technology(s) do you use in addition to Python?",
        name="tech_anaconda",
        converter=notNA
    ),

    ColumnDescription(
        full_name="ORM (SQLAlchemy, PonyORM, etc.):What additional technology(s) do you use in addition to Python?",
        name="tech_orm",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Docker:What additional technology(s) do you use in addition to Python?",
        name="tech_docker",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Vagrant:What additional technology(s) do you use in addition to Python?",
        name="tech_vagrant",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Cloud platforms (Google App Engine, AWS, RackSpace, Heroku and similar):What additional technology(s) do you use in addition to Python?",
        name="tech_cloud",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Hadoop, Spark, etc.:What additional technology(s) do you use in addition to Python?",
        name="tech_hadoop",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Sphinx:What additional technology(s) do you use in addition to Python?",
        name="tech_sphinx",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Buildout:What additional technology(s) do you use in addition to Python?",
        name="tech_buildout",
        converter=notNA
    ),

    ColumnDescription(
        full_name="DevOps tools (Ansible, Chef, Puppet, etc.):What additional technology(s) do you use in addition to Python?",
        name="tech_devops",
        converter=notNA
    ),

    ColumnDescription(
        full_name="redis:What additional technology(s) do you use in addition to Python?",
        name="tech_redis",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Other - Write In::What additional technology(s) do you use in addition to Python?",
        name="tech_other",
        usecol=False
    ),

    ColumnDescription(
        full_name="Google App Engine:What cloud platform(s) do you use?",
        name="cloud_google",
        converter=notNA
    ),

    ColumnDescription(
        full_name="AWS:What cloud platform(s) do you use?",
        name="cloud_aws",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Microsoft Azure:What cloud platform(s) do you use?",
        name="cloud_azure",
        converter=notNA
    ),

    ColumnDescription(
        full_name="RackSpace:What cloud platform(s) do you use?",
        name="cloud_rackspace",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Heroku:What cloud platform(s) do you use?",
        name="cloud_heroku",
        converter=notNA
    ),

    ColumnDescription(
        full_name="DigitalOcean:What cloud platform(s) do you use?",
        name="cloud_do",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Other - Write In::What cloud platform(s) do you use?",
        name="cloud_other",
        usecol=False
    ),

    ColumnDescription(
        full_name="PyCharm Professional Edition:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_pycharmpro",
        converter=notNA
    ),

    ColumnDescription(
        full_name="PyCharm Community Edition:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_pycharmce",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Sublime Text:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_sublime",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Vim:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_vim",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Atom:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_atom",
        converter=notNA
    ),

    ColumnDescription(
        full_name="VS Code:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_vscode",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Eclipse + Pydev:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_eclipse",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Aptana:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_aptana",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Jupyter Notebook:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_ipynb",
        converter=notNA
    ),

    ColumnDescription(
        full_name="IntelliJ IDEA:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_intellij",
        converter=notNA
    ),

    ColumnDescription(
        full_name="NotePad++:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_npp",
        converter=notNA
    ),

    ColumnDescription(
        full_name="IDLE:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_idle",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Emacs:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_emacs",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Python Tools for Visual Studio (PTVS):What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_ptvs",
        converter=notNA
    ),

    ColumnDescription(
        full_name="NetBeans:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_netbeans",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Spyder:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_spyder",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Rodeo:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_rodeo",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Gedit:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_gedit",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Ninja-IDE:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_ninja",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Komodo Editor:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_komodo",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Komodo IDE:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_komodoide",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Wing IDE:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_wing",
        converter=notNA
    ),

    ColumnDescription(
        full_name="TextMate:What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_textmate",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Other - Write In::What editor(s)/IDE(s) have you considered for use in your Python development?",
        name="editor_other",
        usecol=False
    ),

    ColumnDescription(
        full_name="What is the main editor you use for your current Python development?",
        name="main_editor",
        dtype=CategoricalDtype()
    ),

    ColumnDescription(
        full_name="use autocompletion  in your editor:When developing in Python, how often do you…?",
        name="usage_autocomplete",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="use the debugger:When developing in Python, how often do you…?",
        name="usage_debugger",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="refactor your code:When developing in Python, how often do you…?",
        name="usage_refactor",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="use VCS:When developing in Python, how often do you…?",
        name="usage_vcs",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="use code linting (programs that analyze code for potential errors):When developing in Python, how often do you…?",
        name="usage_linting",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="use Python virtual environments for your projects:When developing in Python, how often do you…?",
        name="usage_venv",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="use SQL databases :When developing in Python, how often do you…?",
        name="usage_sql",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="use NoSQL databases:When developing in Python, how often do you…?",
        name="usage_nosql",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="run / debug or edit code on remote machines (remote hosts, VMs, etc.):When developing in Python, how often do you…?",
        name="usage_remote",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="use a Python profiler:When developing in Python, how often do you…?",
        name="usage_profiler",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="write tests for your code:When developing in Python, how often do you…?",
        name="usage_tests",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="use code coverage:When developing in Python, how often do you…?",
        name="usage_coverage",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="use optional type hinting:When developing in Python, how often do you…?",
        name="usage_hinting",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="use Continuous Integration tools:When developing in Python, how often do you…?",
        name="usage_ci",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="use Issue Trackers:When developing in Python, how often do you…?",
        name="usage_issue",
        dtype=CATEGORICAL_USAGE
    ),

    ColumnDescription(
        full_name="Do you regularly work on multiple projects at the same time?",
        name="multiple_projects",
        dtype=CategoricalDtype()
    ),

    ColumnDescription(
        # 'question 220' is: What is the main editor you use for your current Python development?

        full_name="How did you first learn about [question('value'), id='220']?",
        name="maineditor_firstlearned",
        dtype=CategoricalDtype()
    ),

    ColumnDescription(
        full_name="How often do you use [question('value'), id='220']?",
        name="maineditor_usage",
        dtype=CategoricalDtype(
            ["Less frequently", "Monthly", "Weekly", "Daily"],
            ordered=True
        )
    ),

    ColumnDescription(
        full_name="Are you missing any features in [question('value'), id='220']?",
        name="maineditor_missingfeatures",
        converter=lambda x: x == 'Yes'
    ),

    ColumnDescription(
        full_name="How likely is it that you would recommend [question('value'), id='220'] to a friend or colleague?",
        name="maineditor_recommend",
        dtype=np.float32  # int doesn't support NaN
    ),

    ColumnDescription(
        full_name="What do you think is the ratio of these two numbers?:Please think about the total number of Python Web Developers in the world and the total number of Data Scientists using Python.",
        name="webdev_science_ratio_self",
        dtype=CategoricalDtype(
            ["10:1", "5:1", "2:1", "1:1", "1:2", "1:5", "1:10"],
            ordered=True
        )
    ),

    ColumnDescription(
        full_name="What do you think would be the most popular opinion?:Please think about the total number of Python Web Developers in the world and the total number of Data Scientists using Python.",
        name="webdev_science_ratio_others",
        dtype=CategoricalDtype(
            ["10:1", "5:1", "2:1", "1:1", "1:2", "1:5", "1:10"],
            ordered=True
        )
    ),

    ColumnDescription(
        full_name="Most of the time, do you...?",
        name="independent_team_consultant",
        dtype=CategoricalDtype()
    ),

    ColumnDescription(
        full_name="How many people are in your project team?",
        name="team_size",
        dtype=CategoricalDtype(
            [
                "2-7 people",
                "8-12 people",
                "13-20 people",
                "21-40 people",
                "More than 40 people"
            ],
            ordered=True
        )
    ),

    ColumnDescription(
        full_name="What is your employment status?",
        name="employment_status",
        dtype=CategoricalDtype()
    ),

    ColumnDescription(
        full_name="Choose one from the list::Which of the following industries best describes your company's business?",
        name="company_industry",
        dtype=CategoricalDtype()
    ),

    ColumnDescription(
        full_name="Choose one from the list::Which of the following industries do you develop for?",
        name="company_customer_industry",
        dtype=CategoricalDtype()
    ),

    ColumnDescription(
        full_name="How long have you been working in the IT industry?",
        name="years_in_it",
        dtype=CategoricalDtype(
            [
                "Less than 1 year",
                "1 - 2 years",
                "3 - 5 years",
                "6 - 10 years",
                "11+ years"
            ],
            ordered=True
        )
    ),

    ColumnDescription(
        full_name="DBA:Which of the following best describes your job role(s)?",
        name="jobrole_dba",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Architect:Which of the following best describes your job role(s)?",
        name="jobrole_arch",
        converter=notNA
    ),

    ColumnDescription(
        full_name="QA engineer:Which of the following best describes your job role(s)?",
        name="jobrole_qa",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Developer / Programmer:Which of the following best describes your job role(s)?",
        name="jobrole_dev",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Technical support:Which of the following best describes your job role(s)?",
        name="jobrole_support",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Data analyst:Which of the following best describes your job role(s)?",
        name="jobrole_data_analyst",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Business analyst:Which of the following best describes your job role(s)?",
        name="jobrole_ba",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Team lead:Which of the following best describes your job role(s)?",
        name="jobrole_lead",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Product manager:Which of the following best describes your job role(s)?",
        name="jobrole_pm",
        converter=notNA
    ),

    ColumnDescription(
        full_name="CIO / CEO / CTO:Which of the following best describes your job role(s)?",
        name="jobrole_clevel",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Systems analyst:Which of the following best describes your job role(s)?",
        name="jobrole_sysanalyst",
        converter=notNA
    ),

    ColumnDescription(
        full_name="Other - Write In::Which of the following best describes your job role(s)?",
        name="jobrole_other",
        usecol=False
    ),

    ColumnDescription(
        full_name="Could you tell us your age range?",
        name="age",
        dtype=CategoricalDtype(
            [
                "17 or younger",
                "18-20",
                "21-29",
                "30-39",
                "40-49",
                "50-59",
                "60 or older"
            ],
            ordered=True
        )
    ),

    ColumnDescription(
        full_name="What country do you live in?",
        name="country",
        dtype=CategoricalDtype()
    ),

]