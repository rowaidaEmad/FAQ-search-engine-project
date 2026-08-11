"""Streamlit interface for the Banking77 question search engine."""

import streamlit as st

from search_engine import hybrid_search, semantic_search, tfidf_search


st.set_page_config(page_title="Similar Question Search", page_icon="🔎")

st.title("Similar Question Search")
st.write(
    "Find the most similar Banking77 questions using lexical, semantic, "
    "or hybrid retrieval."
)

query = st.text_area(
    "Enter your question",
    placeholder="For example: Where is the card that I ordered?",
)
top_k = st.slider("Number of similar questions", 1, 20, 5)
method = st.radio(
    "Search method",
    ["Lexical", "Semantic", "Hybrid"],
    horizontal=True,
)

if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Please enter a question before searching.")
    else:
        search_functions = {
            "Lexical": tfidf_search,
            "Semantic": semantic_search,
            "Hybrid": hybrid_search,
        }

        try:
            with st.spinner("Preparing the search engine and finding matches..."):
                results = search_functions[method](query, top_k=top_k)

            st.subheader(f"Search method: {method}")
            display_results = results[["rank", "text", "category", "score"]].rename(
                columns={
                    "rank": "Rank",
                    "text": "Similar Question",
                    "category": "Category",
                    "score": "Score",
                }
            )

            st.dataframe(
                display_results,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Rank": st.column_config.NumberColumn(format="%d"),
                    "Score": st.column_config.NumberColumn(format="%.4f"),
                },
            )
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error(
                "The search engine could not be initialized. "
                "Check your internet connection on the first run and try again."
            )
            with st.expander("Technical details"):
                st.code(str(error))
