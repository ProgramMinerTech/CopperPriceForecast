import streamlit as st
import matplotlib.pyplot as plt
from model import table_view

def main():
    st.title("Copper Price Forecast")
    st.header("Data Overview")
    st.dataframe(table_view)

def graph():
    fig, ax = plt.subplots()
    ax.plot(table_view["date"], table_view["LME Copper Cash-Settlement"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Copper Price")
    ax.set_title("Copper Price Trend Over Time")
    ax.legend()
    ax.grid(True)

if __name__ == "__main__":
    main()
    graph()