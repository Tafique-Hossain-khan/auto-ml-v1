import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os

# Test API key configuration
st.title("🔧 Chat Agent Diagnostic Test")

st.markdown("## 1. Configuration Check")

# Test 1: Import dependencies
st.subheader("📦 Dependency Check")
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    st.success("✅ langchain-google-genai imported successfully")
    GEMINI_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ langchain-google-genai import failed: {e}")
    st.code("pip install langchain-google-genai")
    GEMINI_AVAILABLE = False

# Test 2: Check config file
st.subheader("🔧 Configuration Check")
try:
    from config.settings import CHAT_WITH_DATA_CONFIG
    st.write("**CHAT_WITH_DATA_CONFIG found:**")

    # Show config but hide API key
    config_display = dict(CHAT_WITH_DATA_CONFIG)
    if 'api_key' in config_display:
        api_key = config_display['api_key']
        if api_key:
            config_display['api_key'] = f"{'*' * 10}...{api_key[-4:]}" if len(api_key) > 10 else "SET"
        else:
            config_display['api_key'] = "NOT SET"

    st.json(config_display)

    api_key = CHAT_WITH_DATA_CONFIG.get("api_key")
    if not api_key:
        st.error("❌ API key is NOT SET in config")
        st.markdown("**Fix:** Add your Gemini API key to `config/settings.py`")
        st.code('''CHAT_WITH_DATA_CONFIG = {
    "api_key": "your_gemini_api_key_here",  # Get from https://makersuite.google.com/app/apikey
    "model": "gemini-pro",
    "temperature": 0.1,
    "max_output_tokens": 512
}''')
    else:
        st.success(f"✅ API key is set ({len(api_key)} characters)")

except ImportError as e:
    st.error(f"❌ Config import failed: {e}")
    st.markdown("**Fix:** Ensure your `config/settings.py` file exists with CHAT_WITH_DATA_CONFIG")

# Test 3: Test Gemini initialization
st.subheader("🤖 Gemini Initialization Test")

if GEMINI_AVAILABLE:
    try:
        from config.settings import CHAT_WITH_DATA_CONFIG
        api_key = CHAT_WITH_DATA_CONFIG.get("api_key")

        if api_key:
            # Test Gemini connection
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.1,
                max_output_tokens=512
            )

            # Try a simple test call
            from langchain.schema import HumanMessage, SystemMessage

            with st.spinner("Testing Gemini API connection..."):
                test_message = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content="Say 'Hello, API test successful!'")
                ]

                response = llm.invoke(test_message)
                st.success("✅ Gemini API connection successful!")
                st.info(f"Response: {response.content}")

        else:
            st.warning("⚠️ Cannot test - API key not set")

    except Exception as e:
        st.error(f"❌ Gemini initialization failed: {e}")
        st.markdown("**Common issues:**")
        st.markdown("- Invalid API key")
        st.markdown("- API key not active")
        st.markdown("- Network connection issues")
        st.markdown("- API quota exceeded")
else:
    st.warning("⚠️ Cannot test - langchain-google-genai not available")

# Test 4: Chat Agent Creation Test
st.subheader("💬 Chat Agent Creation Test")

# Create sample data
sample_data = pd.DataFrame({
    'age': np.random.randint(18, 80, 100),
    'sex': np.random.choice(['Male', 'Female'], 100),
    'smoking': np.random.choice(['Yes', 'No'], 100),
    'salary': np.random.randint(30000, 150000, 100)
})

st.write("**Sample Dataset:**")
st.dataframe(sample_data.head())

try:
    from components.chat_agent import create_chat_agent, TrulyDynamicChatAgent

    # Create chat agent
    agent = create_chat_agent(sample_data)

    st.success("✅ Chat agent created successfully")

    # Check if AI is available
    if agent.llm is not None:
        st.success("✅ Chat agent has working AI connection")
    else:
        st.warning("⚠️ Chat agent created but AI not available - will use fallbacks")

    # Test a simple query
    st.subheader("🧪 Test Query")

    if st.button("Test Query: 'plot the distribution of age'"):
        with st.spinner("Processing test query..."):
            response = agent.chat("plot the distribution of age")

            st.write("**Response:**")
            st.write(response.get("text", "No text response"))

            chart = response.get("chart")
            if chart:
                st.success("✅ Chart generated successfully!")
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.warning("⚠️ No chart generated")

except Exception as e:
    st.error(f"❌ Chat agent creation failed: {e}")

# Test 5: Manual API Key Input
st.subheader("🔑 Manual API Key Test")
st.markdown("If the above tests fail, you can test with your API key directly:")

manual_key = st.text_input("Enter your Gemini API key:", type="password")

if st.button("Test Manual Key") and manual_key:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=manual_key,
            temperature=0.1,
            max_output_tokens=512
        )

        from langchain.schema import HumanMessage, SystemMessage

        with st.spinner("Testing manual API key..."):
            test_message = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Respond with 'Manual API key test successful!'")
            ]

            response = llm.invoke(test_message)
            st.success("✅ Manual API key test successful!")
            st.info(f"Response: {response.content}")

            st.markdown("**✅ Your API key works! Update your `config/settings.py`:**")
            st.code(f'''CHAT_WITH_DATA_CONFIG = {{
    "api_key": "{manual_key}",
    "model": "gemini-pro", 
    "temperature": 0.1,
    "max_output_tokens": 512
}}''')

    except Exception as e:
        st.error(f"❌ Manual API key test failed: {e}")
        st.markdown("**Possible solutions:**")
        st.markdown("- Get a new API key from https://makersuite.google.com/app/apikey")
        st.markdown("- Check if your key is active and has quota")
        st.markdown("- Verify network connection")

# Summary and fixes
st.markdown("---")
st.subheader("🔧 Quick Fixes")

st.markdown("""
**If tests above failed, try these fixes:**

1. **Install dependencies:**
   ```bash
   pip install langchain-google-genai
   ```

2. **Get API key:**
   - Go to: https://makersuite.google.com/app/apikey
   - Create a new API key
   - Copy the key

3. **Update config/settings.py:**
   ```python
   CHAT_WITH_DATA_CONFIG = {
       "api_key": "your_actual_api_key_here",
       "model": "gemini-pro",
       "temperature": 0.1,
       "max_output_tokens": 512,
       "top_p": 0.8,
       "top_k": 40
   }
   ```

4. **Restart your app:**
   ```bash
   streamlit run app.py
   ```
""")
