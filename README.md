<h1 align="center">
  <img src="https://raw.githubusercontent.com/ak-abhilash/InsightCat/refs/heads/main/logo.png" alt="InsightCat Logo" width="80">
  <br>
  InsightCat
</h1>
<p align="center">Your AI-powered data sidekick for instant insights, quality checks, and smart visualizations. Just upload your data and let InsightCat do the rest.</p>
<p align="center">
  <img src="https://img.shields.io/badge/building%20in-public-brightgreen" alt="build in public">
  <img src="https://img.shields.io/github/license/ak-abhilash/InsightCat" alt="License">
  <img src="https://img.shields.io/github/last-commit/ak-abhilash/InsightCat" alt="Last Commit">
</p>

---

## 💡 What is InsightCat?

InsightCat is an open-source tool that gives you **instant insights from your CSV, Excel, or JSON files.** No coding required. Just upload a file and get:

- 🔍 **AI-powered insights** - Smart analysis with actionable recommendations
- 📊 **Auto-generated charts** - Histograms, bar charts, and distributions 
- 🧹 **Data quality checks** - Missing values, duplicates, and quality scores
- 📈 **Smart visualizations** - Only creates charts that make sense for your data
- 💾 **Download everything** - Save charts and analysis reports

---

## 🔧 Tech Stack

- **Backend:** Python + FastAPI + pandas
- **Frontend:** React + TailwindCSS + Lucide Icons
- **AI Engine:** OpenRouter API (GPT-4o-mini)
- **Charts:** Matplotlib + Seaborn
- **Hosting:** Vercel (frontend), Render (backend)

---

## 📦 How to Use

1. Upload CSV, Excel, or JSON file
2. Get instant data quality assessment (0-100% score)
3. Read AI-generated insights and recommendations  
4. Browse smart auto-generated charts
5. Download charts or full reports

---

## Setup Instructions

### API Keys Setup
1. Register for an account at [OpenRouter](https://openrouter.ai/)
2. Generate an API key from your account dashboard
3. Create a `.env` file by copying `.env.example`:
   ```bash
   cp .env.example .env
   ```
4. Add your API key:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```

### Run Locally
```bash
# Backend
pip install -r requirements.txt
python main.py

# Frontend  
npm install
npm start
```

---

## 🌍 Join the Mission

This is an open-source project to help data learners, educators, and developers explore data quickly.

### Want to contribute?
1. Fork the repo
2. Clone locally  
3. Submit PRs with your ideas

**Ideas welcome:** New chart types, better AI prompts, UI improvements, more file formats

---

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

* Built with ❤️ for the open-source community
* Powered by OpenRouter and modern web technologies
* Inspired by the need for accessible data analysis tools


<p align="center">
  <b>Ready to unlock insights from your data? Give InsightCat a try!</b><br>
  <a href="https://insight-cat.vercel.app">🚀 Try InsightCat Now</a>
</p>
