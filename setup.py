"""
Setup configuration for Task Manager Agent package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="task-manager-agent",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A LangGraph-based recursive task orchestration system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/task-manager",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "langgraph>=0.0.1",
        "langchain-anthropic>=0.1.0",
        "langchain-core>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
)
