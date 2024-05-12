import urllib, urllib.request
from xml.dom import minidom

from interface import abstractiveS

import streamlit as st

# def fetch_articles(topic):
#     # Function to fetch articles based on the topic
#     # You can implement this function to fetch articles from a database or an API
#     # For simplicity, let's just return dummy data
#     articles = [
#         {"title": "Article 1", "link": "https://www.example.com/article1"},
#         {"title": "Article 2", "link": "https://www.example.com/article2"},
#         {"title": "Article 3", "link": "https://www.example.com/article3"},
#         {"title": "Article 4", "link": "https://www.example.com/article4"},
#         {"title": "Article 5", "link": "https://www.example.com/article5"}
#     ]
#     articles=[]
#     return articles

def main():
    articles=[]
    st.title("arXiv Paper Finder")

    st.markdown(
        """
        &copy; Under partial fulfilment of requirements of PROJCS801 by Abhirup Mazumder and Debjyoti Ghosh
        """
    )

    # Input field for entering the topic
    search = st.text_input("Enter the topic of interest:")

    # Button to submit the topic
    if st.button("Submit"):
        if search:
            # Fetch articles based on the entered topic
            query = search.replace(" ", "+")
            url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5'
            data = urllib.request.urlopen(url)
            mytree = minidom.parseString(data.read().decode('utf-8'))
            entry = mytree.getElementsByTagName('entry')
            progress=0.0
            progress_bar = st.progress(progress)
            progress_placeholder = st.empty()
            progress_placeholder.text("We are searching for papers....")
            for y in entry:
                published = y.getElementsByTagName('published')[0]
                title = y.getElementsByTagName('title')[0]
                author = y.getElementsByTagName('author')
                summary = y.getElementsByTagName('summary')[0]
                authors = ''
                for x in author:
                    a_name = x.getElementsByTagName('name')[0]
                    authors = authors + (a_name.firstChild.data) + ', '
                authors = authors[:-2]
                link = y.getElementsByTagName('link')[0]
                link1 = link.attributes['href'].value
                link2 = y.getElementsByTagName('link')[1]
                link3 = link2.attributes['href'].value
                print(title.firstChild.data)
                print(summary.firstChild.data)
                print()
                summ=abstractiveS(summary.firstChild.data)

                article={"title":f"{title.firstChild.data}","abstract":f"{summ}", "link":f"{link1}"}
                articles.append(article)
                progress+=0.20
                progress_bar.progress(progress)

            st.write(f"Here are papers related to '{search}':")
            # Display articles as cards
            for article in articles:
                st.subheader(f"{article['title']}")
                st.write(f"**{article['abstract']}**")
                st.write(article['link'])
                st.write("---")

            progress_bar.empty()
            progress_placeholder.empty()

if __name__ == "__main__":
    main()




