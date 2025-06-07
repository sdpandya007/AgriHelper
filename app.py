import streamlit as st
import pandas as pd
import joblib
import requests
import os
import time
from groq import Groq
import plotly.express as px
import re
from deep_translator import GoogleTranslator
import speech_recognition as sr  


# Load models
crop_model = joblib.load('data/trained_crop_model.pkl')
fertilizer_model = joblib.load('data/trained_fertilizer_model.pkl')

# WeatherAPI Key
WEATHER_API_KEY = "77d2f53d912c472b89b170125252102"
WEATHER_BASE_URL = "http://api.weatherapi.com/v1/forecast.json"

# Initialize Groq client
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables")
client = Groq(api_key=GROQ_API_KEY)

# Translation function
def translate(text, target_lang):
    if not text or target_lang == 'en':
        return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Function to capture voice input
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write(translate("Listening... Please speak.", target_lang))
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(translate(f"You said: {text}", target_lang))
            return text
        except sr.UnknownValueError:
            st.error(translate("Could not understand audio", target_lang))
            return ""
        except sr.RequestError as e:
            st.error(translate(f"Speech recognition error: {e}", target_lang))
            return ""

# Language configuration
LANGUAGES = {
    "English": "en",
    "हिंदी": "hi",
    "ગુજરાતી": "gu"
}

# Sidebar setup
st.sidebar.markdown("""
<div style="text-align:center;padding:20px;background-color:#f0f8ff;border-radius:10px;">
    <h2 style="color:#333;">🌾 Kishan Sathi </h2>
</div>
""", unsafe_allow_html=True)
selected_lang = st.sidebar.selectbox(
    translate("Choose language", "en"),
    list(LANGUAGES.keys())
)
target_lang = LANGUAGES[selected_lang]

# Feature mapping for multilingual support
feature_options = {
    "HomeAs": "HomeAs",
    "CropFertilizer": "Crop & Fertilizer Recommendation",
    "Weather": "Weather Forecast",
    "Schemes": "Government Schemes",
    "Chatbot": "Chatbot Support",
    "MarketPrices": "Market Price Insights",
    "LoanCalculator": "Loan & Subsidy Calculator",
    "About": "About"
}
translated_features = {k: translate(v, target_lang) for k, v in feature_options.items()}
app_mode = st.sidebar.selectbox(
    translate("Choose a feature", target_lang),
    list(translated_features.values()),
    index=list(feature_options.keys()).index("HomeAs")
)
app_mode_key = list(feature_options.keys())[list(translated_features.values()).index(app_mode)]

# -------------------------------------------------------------------
# Function to interact with Groq API for chatbot
# -------------------------------------------------------------------
def get_chatbot_response(prompt, target_lang):
    # Translate user query to English for model processing
    english_prompt = translate(prompt, "en")
    system_prompt = (
        "You are an agricultural expert. "
        "Provide concise, actionable advice. "
        "Answer only farming-related questions. "
        f"Question: {english_prompt}\n"
        "Answer:"
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Agricultural expert system"},
                {"role": "user", "content": system_prompt}
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=900,
            temperature=0.7
        )
        # Translate response back to user's language
        response = chat_completion.choices[0].message.content.strip()
        return translate(response, target_lang)
    except Exception as e:
        st.error(translate(f"API Error: {e}", target_lang))
        return translate("Request failed. Try again later.", target_lang)


# -------------------------------------------------------------------
# Home Page
# -------------------------------------------------------------------
if app_mode_key == "HomeAs":
    st.markdown(f"""
    <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;margin-bottom:20px;">
        <h1 style="color:#333;text-align:center;">{translate('Welcome to Kishan Sathi 🌱', target_lang)}</h1>
        <p style="font-size:18px;color:#555;text-align:center;">
            {translate('Your one-stop solution for smart farming. Explore features like crop recommendations, fertilizer suggestions, weather forecasts, government schemes, and AI-powered farming advice.', target_lang)}
        </p>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# Market Price Insights
# -------------------------------------------------------------------

elif app_mode_key == "MarketPrices":
    st.markdown(f"""
    <div style="background-color:#d1e7dd;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h2 style="color:#0f5132;">{translate('Market Price Insights 📊', target_lang)}</h2>
        <p style="color:#0f5132;">{translate('Get accurate price trends for your region.', target_lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Indian states and districts mapping (Top agricultural states)
    STATES = {
        "Andhra Pradesh": ["Anantapur", "Guntur", "Visakhapatnam"],
        "Gujarat": ["Ahmedabad", "Rajkot", "Surat"],
        "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
        "Punjab": ["Ludhiana", "Amritsar", "Patiala"],
        "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi"],
        "Karnataka": ["Bengaluru", "Mysuru", "Hubli"],
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"]
    }
    
    # Input parameters with Indian context
    col1, col2 = st.columns(2)
    with col1:
        market_crop = st.text_input(translate("Crop Name", target_lang), "")
        
        state = st.selectbox(
            translate("State", target_lang),
            list(STATES.keys())
        )
        
        market_type = st.selectbox(
            translate("Market Type", target_lang),
            [translate("Wholesale", target_lang), 
             translate("Retail", target_lang),
             translate("Mandi", target_lang)]
        )
        
    with col2:
        district = st.selectbox(
            translate("District", target_lang),
            STATES.get(state, [])
        )
        
        commodity_grade = st.selectbox(
            translate("Commodity Grade", target_lang),
            [translate("Grade A", target_lang), 
             translate("Grade B", target_lang),
             translate("Organic", target_lang)]
        )
        
        analysis_type = st.radio(
            translate("Analysis Type", target_lang),
            [translate("Current Prices", target_lang), 
             translate("Trend Analysis", target_lang),
             translate("Price Forecast", target_lang)]
        )

    if st.button(translate("Get Market Insights 📈", target_lang)):
        if not all([market_crop, state, district]):
            st.warning(translate("Please enter crop, state and district", target_lang))
        else:
            # Enhanced prompt with Indian agricultural context
            prompt = (
                f"Provide a detailed market analysis report for {market_crop} in {district}, {state} including:\n"
                f"- Current {market_type} price range (per quintal) for {commodity_grade} grade\n"
                f"- Comparison with neighboring districts\n"
                f"- MSP (Minimum Support Price) information\n"
                f"- Government procurement centers in {district}\n"
                f"- Transportation cost estimates to nearest mandi\n"
                f"- {analysis_type} with historical data comparison\n"
                f"- Impact of recent government policies\n"
                f"- Festive season price variations\n"
                f"- Recommendations for small/marginal farmers\n"
                f"Format: Clear sections with bullet points and ₹ values"
            )
            
            with st.spinner(translate("Analyzing market data...", target_lang)):
                market_report = get_chatbot_response(prompt, target_lang)
            
            # Display report with Indian context
            st.subheader(translate("Market Analysis Report", target_lang))
            st.markdown(f"""
            <div style="background-color:#f9f9f9;padding:15px;border-radius:10px;margin-bottom:20px;color:#333;">
                <h3>{translate('Price Insights for', target_lang)} {market_crop} in {district}</h3>
                <p>{market_report}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced visualization parsing
            try:
                # Price range parsing
                price_pattern = r'₹(\d+,\d+)-₹(\d+,\d+)\s*per quintal'
                price_match = re.search(price_pattern, market_report)
                
                # MSP parsing
                msp_pattern = r'MSP:\s*₹(\d+,\d+)'
                msp_match = re.search(msp_pattern, market_report)
                
                # Create price comparison chart
                if price_match and msp_match:
                    min_price = float(price_match.group(1).replace(',', ''))
                    max_price = float(price_match.group(2).replace(',', ''))
                    msp_price = float(msp_match.group(1).replace(',', ''))
                    
                    price_df = pd.DataFrame({
                        "Price Type": [
                            translate("Minimum Market Price", target_lang),
                            translate("Maximum Market Price", target_lang),
                            translate("MSP", target_lang)
                        ],
                        "Price (₹/quintal)": [min_price, max_price, msp_price]
                    })
                    
                    fig_price = px.bar(
                        price_df,
                        x="Price Type",
                        y="Price (₹/quintal)",
                        title=translate("Price Comparison", target_lang),
                        color="Price Type",
                        color_discrete_map={
                            translate("MSP", target_lang): "#28a745",
                            translate("Minimum Market Price", target_lang): "#ffc107",
                            translate("Maximum Market Price", target_lang): "#dc3545"
                        }
                    )
                    st.plotly_chart(fig_price, use_container_width=True)
                
                # District comparison parsing
                district_pattern = r'(\w+ district):\s*₹([\d,]+)'
                district_matches = re.findall(district_pattern, market_report)
                
                if district_matches:
                    districts = []
                    prices = []
                    for match in district_matches:
                        districts.append(match[0])
                        prices.append(float(match[1].replace(',', '')))
                    
                    comp_df = pd.DataFrame({
                        translate("District", target_lang): districts,
                        translate("Price (₹/quintal)", target_lang): prices
                    })
                    
                    fig_comp = px.bar(
                        comp_df,
                        x=translate("District", target_lang),
                        y=translate("Price (₹/quintal)", target_lang),
                        title=translate("Neighboring District Comparison", target_lang),
                        color=translate("District", target_lang)
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                
            except Exception as e:
                st.warning(translate(f"Data parsing error: {e}", target_lang))
# -------------------------------------------------------------------
# Loan & Subsidy Calculator
# -------------------------------------------------------------------
elif app_mode_key == "LoanCalculator":
    st.markdown(f"""
    <div style="background-color:#d4edda;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h2 style="color:#155724;">{translate('Loan & Subsidy Calculator 💰', target_lang)}</h2>
        <p style="color:#155724;">{translate('Calculate your loan EMI and subsidy benefits.', target_lang)}</p>
    </div>
    """, unsafe_allow_html=True)

    loan_amount = st.number_input(translate("Loan Amount (₹)", target_lang), 0, 10000000, 100000)
    interest_rate = st.number_input(translate("Annual Interest Rate (%)", target_lang), 0.0, 20.0, 8.0)
    tenure_years = st.number_input(translate("Loan Tenure (Years)", target_lang), 1, 30, 5)
    subsidy_percentage = st.number_input(translate("Subsidy Percentage (%)", target_lang), 0.0, 100.0, 25.0)

    if st.button(translate("Calculate EMI & Subsidy", target_lang)):
        monthly_interest_rate = interest_rate / 1200
        tenure_months = tenure_years * 12
        emi = (loan_amount * monthly_interest_rate * (1 + monthly_interest_rate) ** tenure_months) / \
              ((1 + monthly_interest_rate) ** tenure_months - 1)
        subsidy_amount = (subsidy_percentage / 100) * loan_amount
        net_loan_amount = loan_amount - subsidy_amount

        st.subheader(translate("Loan Details", target_lang))
        st.metric(translate("Monthly EMI", target_lang), f"₹{emi:.2f}")
        st.metric(translate("Subsidy Amount", target_lang), f"₹{subsidy_amount:.2f}")
        st.metric(translate("Net Loan Amount", target_lang), f"₹{net_loan_amount:.2f}")




# -------------------------------------------------------------------
# Crop & Fertilizer Recommendation
# -------------------------------------------------------------------
elif app_mode_key == "CropFertilizer":
    st.markdown(f"""
    <div style="background-color:#fff3cd;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h2 style="color:#856404;">{translate('Crop & Fertilizer Advisor 🌱', target_lang)}</h2>
        <p style="color:#856404;">{translate('Enter your soil parameters below to get personalized recommendations.', target_lang)}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input(translate("Nitrogen (N)", target_lang), 0, 150, 90)
        P = st.number_input(translate("Phosphorus (P)", target_lang), 0, 150, 42)
        K = st.number_input(translate("Potassium (K)", target_lang), 0, 150, 43)
        temp = st.number_input(translate("Temperature (°C)", target_lang), 0.0, 50.0, 20.88)
    with col2:
        humidity = st.number_input(translate("Humidity (%)", target_lang), 0.0, 100.0, 82.0)
        ph = st.number_input(translate("Soil pH", target_lang), 0.0, 14.0, 6.5)
        rainfall = st.number_input(translate("Rainfall (mm)", target_lang), 0.0, 300.0, 202.94)

    if st.button(translate("Get Recommendations 🌱", target_lang)):
        # Crop prediction
        crop_input = pd.DataFrame({
            'N': [N], 'P': [P], 'K': [K],
            'temperature': [temp],
            'humidity': [humidity],
            'ph': [ph], 
            'rainfall': [rainfall]
        })
        crop_pred = crop_model.predict(crop_input)[0]
        # Fertilizer prediction
        fert_input = pd.DataFrame({'Nitrogen': [N], 'Potassium': [K], 'Phosphorous': [P]})
        fert_pred = fertilizer_model.predict(fert_input)[0]
        # Translate results
        crop_trans = translate(crop_pred.upper(), target_lang)
        fert_trans = translate(fert_pred, target_lang)
        st.success(f"🌱 {translate('Recommended Crop:', target_lang)} **{crop_trans}**")
        st.success(f"🌿 {translate('Recommended Fertilizer:', target_lang)} **{fert_trans}**")
        st.balloons()

        # Visualization of Input Parameters
        st.subheader(translate("Input Parameters Overview", target_lang))
        input_df = pd.DataFrame({
            "Parameters": [
                translate("Nitrogen (N)", target_lang),
                translate("Phosphorus (P)", target_lang),
                translate("Potassium (K)", target_lang),
                translate("Temperature (°C)", target_lang),
                translate("Humidity (%)", target_lang),
                translate("Soil pH", target_lang),
                translate("Rainfall (mm)", target_lang)
            ],
            "Values": [N, P, K, temp, humidity, ph, rainfall]
        })
        fig = px.bar(
            input_df,
            x="Parameters",
            y="Values",
            title=translate("Input Parameters Overview", target_lang),
            labels={
                "Parameters": translate("Parameter", target_lang),
                "Values": translate("Value", target_lang)
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # Generate Structured Cultivation Report
        report_prompt = (
            f"Create a detailed cultivation guide for {crop_pred} with:\n"
            f"- Soil preparation steps\n"
            f"- Sowing guidelines (spacing, depth, time)\n"
            f"- Irrigation schedule and water requirements\n"
            f"- {fert_pred} fertilizer application timing and quantity\n"
            f"- Pest and disease control measures\n"
            f"- Expected yield timeline\n"
            f"- Harvesting tips\n"
            f"- Profit maximization strategies\n"
            f"- Effects of weather conditions on growth\n"
        )
        with st.spinner(translate("Generating detailed report...", target_lang)):
            report = get_chatbot_response(report_prompt, target_lang)
        
        st.subheader(translate("Full Cultivation Guide", target_lang))
        st.markdown(f"""
        <div style="background-color:#f9f9f9;padding:15px;border-radius:10px;margin-bottom:20px;color:#333;">
            <h3>{translate('Detailed Report', target_lang)}</h3>
            <p>{report}</p>
        </div>
        """, unsafe_allow_html=True)

        # Parse Metrics from English Response for Visualization
        try:
            english_report = translate(report, "en")
            yield_match = re.search(r'(\d+\.?\d*)\s*kg/ha', english_report)
            pest_match = re.search(r'(\d+\.?\d*)%', english_report)
            irrigation_match = re.search(r'(\d+\.?\d*)\s*(liters|L)', english_report)
            profit_margin_match = re.search(r'(\d+\.?\d*)\s*(USD|INR|Rs)', english_report)  # New metric

            # Yield Estimate
            if yield_match:
                st.subheader(translate("Yield Estimate", target_lang))
                st.metric(
                    translate("Expected Yield", target_lang),
                    f"{yield_match.group(1)} kg/ha",
                    delta=None
                )

            # Profit Margin Analysis
            if profit_margin_match:
                st.subheader(translate("Profit Margin Analysis", target_lang))
                st.metric(
                    translate("Estimated Profit", target_lang),
                    f"{profit_margin_match.group(1)} {profit_margin_match.group(2)}"
                )

            # Pest Risk Progress Bar
            if pest_match:
                st.subheader(translate("Pest Risk Analysis", target_lang))
                st.progress(int(float(pest_match.group(1))))
                st.caption(f"{pest_match.group(1)}% {translate('Risk', target_lang)}")

            # Irrigation Requirement Chart
            if irrigation_match:
                st.subheader(translate("Irrigation Requirements", target_lang))
                irrigation_data = {
                    translate("Irrigation Type", target_lang): ["Drip Irrigation", "Sprinkler Irrigation", "Flood Irrigation"],
                    translate("Water Usage (Liters)", target_lang): [float(irrigation_match.group(1)), float(irrigation_match.group(1)) * 1.2, float(irrigation_match.group(1)) * 1.5]
                }
                irrigation_df = pd.DataFrame(irrigation_data)
                fig_irrigation = px.bar(
                    irrigation_df,
                    x=translate("Irrigation Type", target_lang),
                    y=translate("Water Usage (Liters)", target_lang),
                    title=translate("Irrigation Water Usage Comparison", target_lang),
                    labels={
                        translate("Irrigation Type", target_lang): translate("Type", target_lang),
                        translate("Water Usage (Liters)", target_lang): translate("Water Usage", target_lang)
                    }
                )
                st.plotly_chart(fig_irrigation, use_container_width=True)

            # Fertilizer Composition Pie Chart
            st.subheader(translate("Fertilizer Composition", target_lang))
            fertilizer_data = {
                translate("Nutrient", target_lang): [translate("Nitrogen", target_lang), translate("Phosphorus", target_lang), translate("Potassium", target_lang)],
                translate("Amount Applied (kg/ha)", target_lang): [N, P, K]
            }
            fertilizer_df = pd.DataFrame(fertilizer_data)
            fig_fertilizer = px.pie(
                fertilizer_df,
                names=translate("Nutrient", target_lang),
                values=translate("Amount Applied (kg/ha)", target_lang),
                title=translate("Fertilizer Composition", target_lang),
                hole=0.3  # Donut chart
            )
            st.plotly_chart(fig_fertilizer, use_container_width=True)

            # Weather Condition Effects
            st.subheader(translate("Weather Condition Effects", target_lang))
            weather_effects = {
                translate("Condition", target_lang): [
                    translate("High Temperature", target_lang),
                    translate("Low Humidity", target_lang),
                    translate("Excess Rainfall", target_lang)
                ],
                translate("Impact", target_lang): [
                    translate("May cause wilting and reduce yield.", target_lang),
                    translate("Can lead to dry soil and poor nutrient absorption.", target_lang),
                    translate("May result in waterlogging and root rot.", target_lang)
                ]
            }
            weather_df = pd.DataFrame(weather_effects)
            st.table(weather_df)

            # Profit Maximization Tips
            st.subheader(translate("Tips for Maximizing Profit", target_lang))
            profit_tips = [
                translate("Use drip irrigation to conserve water and reduce costs.", target_lang),
                translate("Apply fertilizers at the right time to maximize nutrient absorption.", target_lang),
                translate("Monitor pest risks regularly to prevent crop damage.", target_lang),
                translate("Harvest at the optimal time to ensure maximum yield.", target_lang)
            ]
            for tip in profit_tips:
                st.markdown(f"- {tip}")

        except Exception as e:
            st.warning(translate(f"Parsing error: {e}", target_lang))

# -------------------------------------------------------------------
# Weather Forecast
# -------------------------------------------------------------------
elif app_mode_key == "Weather":
    st.markdown(f"""
    <div style="background-color:#fff3cd;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h2 style="color:#856404;">{translate('Weather Forecast ☁️',target_lang)}</h2>
        <p style="color:#856404;">{translate('Enter your city name to get real-time weather updates.',target_lang)}</p>
    </div>
    """, unsafe_allow_html=True)

    city = st.text_input(translate("Enter City", target_lang), "")
    if st.button(translate("Get Forecast 🌡️", target_lang)):
        try:
            url = f"{WEATHER_BASE_URL}?key={WEATHER_API_KEY}&q={city}&days=5"
            data = requests.get(url).json()
            if "error" not in data:
                current = data["current"]
                forecast = data["forecast"]["forecastday"]
                st.success(f"📝 {translate('Current Weather in', target_lang)} {city.capitalize()}:")
                st.write(f"🌡️ {translate('Temperature:', target_lang)} {current['temp_c']}°C")
                st.write(f"💧 {translate('Humidity:', target_lang)} {current['humidity']}%")
                st.write(f"🌬️ {translate('Wind:', target_lang)} {current['wind_kph']} km/h")
                st.write(f"☁️ {translate('Condition:', target_lang)} {translate(current['condition']['text'], target_lang)}")
                st.subheader(translate("3-Day Forecast:", target_lang))
                for day in forecast[:3]:
                    st.write(f"📅 {day['date']}")
                    st.write(f"☀️ {translate('Max Temp:', target_lang)} {day['day']['maxtemp_c']}°C")
                    st.write(f"🌙 {translate('Min Temp:', target_lang)} {day['day']['mintemp_c']}°C")
                    st.write(f"☁️ {translate('Condition:', target_lang)} {translate(day['day']['condition']['text'], target_lang)}")
                    st.write("---")
            else:
                st.error(translate("City not found", target_lang))
        except Exception as e:
            st.error(translate(f"Error: {e}", target_lang))

# -------------------------------------------------------------------
# Government Schemes
# -------------------------------------------------------------------
elif app_mode_key == "Schemes":
    st.markdown(f"""
    <div style="background-color:#d4edda;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h2 style="color:#155724;">{translate('Government Schemes 🏛️',target_lang)}</h2>
        <p style="color:#155724;">{translate('Search for government schemes related to agriculture.',target_lang)}</p>
    </div>
    """, unsafe_allow_html=True)

    def fetch_schemes_from_api():
        # Fallback data
        static_data = [
            {"S.No.": "1", "Scheme Name": "Pradhan Mantri Fasal Bima Yojana"},
            {"S.No.": "2", "Scheme Name": "Paramparagat Krishi Vikas Yojana"},
            {"S.No.": "3", "Scheme Name": "Soil Health Card Scheme"},
            {"S.No.": "4", "Scheme Name": "National Mission for Sustainable Agriculture"}
        ]
        # Try OGD API first
        try:
            api_url = "https://api.data.gov.in/resource/9afdf346-16d7-4f17-a2e3-684540c59a77"
            params = {
                "api-key": "579b464db66ec23bdd0000015e4e4c86f7d4482c48a11ac8da071ff9",
                "format": "json",
                "offset": 0,
                "limit": 100
            }
            response = requests.get(api_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("records"):
                    schemes = []
                    for item in data["records"]:
                        schemes.append({
                            "S.No.": item.get("s_no_", "N/A"),
                            "Scheme Name": item.get("name_of_mission___scheme", "N/A")
                        })
                    df = pd.DataFrame(schemes)
                    df.to_csv("data/schemes.csv", index=False)
                    return df
        except Exception as e:
            st.warning(translate(f"API Error: {e}. Using fallback data", target_lang))
        # Return static data if API fails
        return pd.DataFrame(static_data)

    def load_schemes():
        os.makedirs("data", exist_ok=True)
        if not os.path.exists("data/schemes.csv") or \
           (time.time() - os.path.getmtime("data/schemes.csv")) > 86400:  # 24h cache
            return fetch_schemes_from_api()
        return pd.read_csv("data/schemes.csv")

    schemes_df = load_schemes()
    search_term = st.text_input(translate("Search Schemes 🔍", target_lang), "")
    filtered = schemes_df[
        schemes_df["Scheme Name"].str.contains(
            search_term, 
            case=False, 
            na=False,
            regex=False
        )
    ] if search_term else schemes_df
    st.subheader(translate("Available Schemes:", target_lang))
    if filtered.empty:
        st.warning(translate("No matching schemes found", target_lang))
    else:
        cols = st.columns(2)
        for idx, (_, row) in enumerate(filtered.iterrows()):
            with cols[idx % 2]:
                with st.expander(f"{row['S.No.']}. {translate(row['Scheme Name'], target_lang)}"):
                    st.markdown(f"""
                    **{translate('Scheme:', target_lang)}**  
                    {translate(row['Scheme Name'], target_lang)}  
                    **{translate('S.No.:', target_lang)}** {row['S.No.']}  
                    """)

# -------------------------------------------------------------------
# Chatbot Support
# -------------------------------------------------------------------
elif app_mode_key == "Chatbot":
    st.markdown(f"""
    <div style="background-color:#fff3cd;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h2 style="color:#856404;">{translate('AI Farming Assistant 🤖',target_lang)}</h2>
        <p style="color:#856404;">{translate('Ask about:',target_lang)}</p>
        <ul style="color:#856404;">
            <li>{translate('Crop selection',target_lang)}</li>
            <li>{translate('Fertilizer usage',target_lang)}</li>
            <li>{translate('Weather advice',target_lang)}</li>
            <li>{translate('Government schemes',target_lang)}</li>
            <li>{translate('Pest control',target_lang)}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Option to choose input type
    # Ask the user to choose input type (define input_type first)
input_type = st.radio(
    translate("Choose input type:", target_lang),
    [translate("Text Input", target_lang)]  # Removed voice input if you're not using it
)

# Now safely use input_type
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

if input_type == translate("Text Input", target_lang):
    st.session_state.user_query = st.text_input(translate("Your question:", target_lang), st.session_state.user_query)

if st.button(translate("Get Answer 🌱", target_lang)):
    if st.session_state.user_query.strip():
        with st.spinner(translate("Processing...", target_lang)):
            response = get_chatbot_response(st.session_state.user_query, target_lang)
        st.success(f"**{translate('Answer:', target_lang)}** {response}")
    else:
        st.warning(translate("Please enter a question", target_lang))



# -------------------------------------------------------------------
# About Page
# -------------------------------------------------------------------
elif app_mode_key == "About":
    # Add custom CSS for styling
    st.markdown("""
    <style>
        /* Container Styling */
        .about-container {
            background: linear-gradient(135deg, #f0f8ff, #e6f7ff);
            padding: 30px;
            border-radius: 20px;
            margin: 20px 0;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        }

        /* Title Styling */
        .about-title {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #2c3e50;
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 20px;
        }

        /* Description Styling */
        .about-description {
            font-size: 18px;
            color: #34495e;
            text-align: justify;
            line-height: 1.6;
            margin-bottom: 30px;
        }

        /* Feature Item Styling */
        .feature-item {
            background: #ffffff;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            font-size: 16px;
            color: #555;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .feature-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
        }

        /* Tech Stack Styling */
        .tech-stack {
            font-size: 16px;
            color: #7f8c8d;
            text-align: center;
            margin-top: 20px;
        }

        /* Footer Styling */
        .footer {
            text-align: center;
            font-size: 16px;
            color: #34495e;
            margin-top: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Start the About Section
    # st.markdown('<div class="about-container">', unsafe_allow_html=True)

    # Title
    st.markdown(f'<p class="about-title about-container">🌱 {translate("About AgriHelper 📜", target_lang)}</p>', unsafe_allow_html=True)

    # Description
    st.markdown(f'<p class="about-description">{translate("AgriHelper is a cutting-edge platform designed to empower farmers with smart solutions.", target_lang)}</p>', unsafe_allow_html=True)

    # Feature List
    features = [
        "🌾 " + translate("Data-driven crop recommendations", target_lang),
        "🌱 " + translate("Personalized fertilizer suggestions", target_lang),
        "☁️ " + translate("Real-time weather forecasts", target_lang),
        "🏛️ " + translate("Government scheme information", target_lang),
        "💬 " + translate("AI-powered chatbot support", target_lang),
        "💰 " + translate("Loan & subsidy calculator", target_lang),
        "📊 " + translate("Market price insights", target_lang),
    ]

    for feature in features:
        st.markdown(f'<div class="feature-item">{feature}</div>', unsafe_allow_html=True)

    # Tech Stack
    st.markdown(f"""
    <p class="tech-stack">
        🛠️ {translate('Built using:', target_lang)} Python 🐍, Streamlit 🚀, Scikit-learn 🧠, WeatherAPI ☁️, Groq API 💡
    </p>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("""
<div style="text-align:center;padding:7px;background-color:#f8f9fa;margin-top:20px;border-radius:10px;">
    <p style="font-size:20px;color:#555;text-align:center;">
    Made with ❤️ by KishanSathi
    </p>
</div>
""", unsafe_allow_html=True)








