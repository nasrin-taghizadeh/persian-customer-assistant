from langchain_groq import ChatGroq
import os
import yfinance as yf
import pandas as pd

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage

from datetime import date
import pandas as pd
import plotly.graph_objects as go

import streamlit as st

os.environ['GROQ_API_KEY'] = "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"

@tool
def get_stock_info(symbol, key):
    '''Return the correct stock info value given the appropriate symbol and key. Infer valid key from the user prompt; it must be one of the following:

    address1, city, state, zip, country, phone, website, industry, industryKey, industryDisp, sector, sectorKey, sectorDisp, longBusinessSummary, fullTimeEmployees, companyOfficers, auditRisk, boardRisk, compensationRisk, shareHolderRightsRisk, overallRisk, governanceEpochDate, compensationAsOfEpochDate, maxAge, priceHint, previousClose, open, dayLow, dayHigh, regularMarketPreviousClose, regularMarketOpen, regularMarketDayLow, regularMarketDayHigh, dividendRate, dividendYield, exDividendDate, beta, trailingPE, forwardPE, volume, regularMarketVolume, averageVolume, averageVolume10days, averageDailyVolume10Day, bid, ask, bidSize, askSize, marketCap, fiftyTwoWeekLow, fiftyTwoWeekHigh, priceToSalesTrailing12Months, fiftyDayAverage, twoHundredDayAverage, currency, enterpriseValue, profitMargins, floatShares, sharesOutstanding, sharesShort, sharesShortPriorMonth, sharesShortPreviousMonthDate, dateShortInterest, sharesPercentSharesOut, heldPercentInsiders, heldPercentInstitutions, shortRatio, shortPercentOfFloat, impliedSharesOutstanding, bookValue, priceToBook, lastFiscalYearEnd, nextFiscalYearEnd, mostRecentQuarter, earningsQuarterlyGrowth, netIncomeToCommon, trailingEps, forwardEps, pegRatio, enterpriseToRevenue, enterpriseToEbitda, 52WeekChange, SandP52WeekChange, lastDividendValue, lastDividendDate, exchange, quoteType, symbol, underlyingSymbol, shortName, longName, firstTradeDateEpochUtc, timeZoneFullName, timeZoneShortName, uuid, messageBoardId, gmtOffSetMilliseconds, currentPrice, targetHighPrice, targetLowPrice, targetMeanPrice, targetMedianPrice, recommendationMean, recommendationKey, numberOfAnalystOpinions, totalCash, totalCashPerShare, ebitda, totalDebt, quickRatio, currentRatio, totalRevenue, debtToEquity, revenuePerShare, returnOnAssets, returnOnEquity, freeCashflow, operatingCashflow, earningsGrowth, revenueGrowth, grossMargins, ebitdaMargins, operatingMargins, financialCurrency, trailingPegRatio
    
    If asked generically for 'stock price', use currentPrice
    '''
    data = yf.Ticker(symbol)
    stock_info = data.info
    return stock_info[key]


@tool
def get_historical_price(symbol, start_date, end_date):
    """
    Fetches historical stock prices for a given symbol from 'start_date' to 'end_date'.
    - symbol (str): Stock ticker symbol.
    - end_date (date): Typically today unless a specific end date is provided. End date MUST be greater than start date
    - start_date (date): Set explicitly, or calculated as 'end_date - date interval' (for example, if prompted 'over the past 6 months', date interval = 6 months so start_date would be 6 months earlier than today's date). Default to '1900-01-01' if vaguely asked for historical price. Start date must always be before the current date
    """

    data = yf.Ticker(symbol)
    hist = data.history(start=start_date, end=end_date)
    hist = hist.reset_index()
    hist[symbol] = hist['Close']
    return hist[['Date', symbol]]

def plot_price_over_time(historical_price_dfs):

    full_df = pd.DataFrame(columns = ['Date'])
    for df in historical_price_dfs:
        full_df = full_df.merge(df, on = 'Date', how = 'outer')

    # Create a Plotly figure
    fig = go.Figure()
    
    # Dynamically add a trace for each stock symbol in the DataFrame
    for column in full_df.columns[1:]:  # Skip the first column since it's the date
        fig.add_trace(go.Scatter(x=full_df['Date'], y=full_df[column], mode='lines+markers', name=column))
    
    
    # Update the layout to add titles and format axis labels
    fig.update_layout(
        title='Stock Price Over Time: ' + ', '.join(full_df.columns.tolist()[1:]),
        xaxis_title='Date',
        yaxis_title='Stock Price (USD)',
        yaxis_tickprefix='$',
        yaxis_tickformat=',.2f',
        xaxis=dict(
            tickangle=-45,
            nticks=20,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=True,   # Enable y-axis grid lines
            gridcolor='lightgrey',  # Set grid line color
        ),
        legend_title_text='Stock Symbol',
        plot_bgcolor='gray',  # Set plot background to white
        paper_bgcolor='gray',  # Set overall figure background to white
        legend=dict(
            bgcolor='gray',  # Optional: Set legend background to white
            bordercolor='black'
        )
    )
    
    # Show the figure
    st.plotly_chart(fig, use_container_width=True)

def call_functions(llm_with_tools, user_prompt):
    system_prompt = 'You are a helpful finance assistant that analyzes stocks and stock prices. Today is {today}'.format(today = date.today())
    
    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    historical_price_dfs = []
    symbols = []
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"get_stock_info": get_stock_info, "get_historical_price": get_historical_price}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        if tool_call['name'] == 'get_historical_price':
            historical_price_dfs.append(tool_output)
            symbols.append(tool_output.columns[1])
        else:
            messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    
    if len(historical_price_dfs) > 0:
        plot_price_over_time(historical_price_dfs)
    
        symbols = ' and '.join(symbols)
        messages.append(ToolMessage('Tell the user that a historical stock price chart for {symbols} been generated.'.format(symbols=symbols), tool_call_id=0))

    return llm_with_tools.invoke(messages).content



def main():

    llm = ChatGroq(groq_api_key = os.getenv('GROQ_API_KEY'),model = 'llama3-70b-8192')
    
    tools = [get_stock_info, get_historical_price]
    llm_with_tools = llm.bind_tools(tools)

    # # Display the Groq logo
    # spacer, col = st.columns([5, 1])
    # with col:
    #     st.image('groqcloud_darkmode.png')
    #
    # # Display the title and introduction of the application
    # st.title("Groqing the Stock Market with Llama 3")
    # multiline_text = """
    # Try to ask it "What is the current price of Meta stock?" or "Show me the historical prices of Apple vs Microsoft stock over the past 6 months.".
    # """
    #
    # st.markdown(multiline_text, unsafe_allow_html=True)
    #
    # # Add customization options to the sidebar
    # st.sidebar.title('Customization')
    # additional_context = st.sidebar.text_input('Enter additional summarization context for the LLM here (i.e. write it in spanish):')
    #
    # # Get the user's question
    # user_question = st.text_input("Ask a question about a stock or multiple stocks:")

    user_question = "Show me the historical prices of Amazon and Tesla over the past year"
    if user_question:
        response = call_functions(llm_with_tools, user_question)
        print(response)
        # st.write(response)



if __name__ == "__main__":
    main()
