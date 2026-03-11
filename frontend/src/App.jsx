import { useState, useEffect } from 'react'
import './App.css'

// API base URL — our FastAPI server
const API = 'http://127.0.0.1:8000'

export default function App() {

  // ── STATE ──────────────────────────────────────────
  // each state variable holds one piece of data
  // when state changes React automatically re-renders

  const [weather, setWeather]       = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [sentiment, setSentiment]   = useState(null)
  const [anomaly, setAnomaly]       = useState(null)
  const [topics, setTopics]         = useState(null)
  const [loading, setLoading]       = useState(true)
  const [question, setQuestion]     = useState('')
  const [answer, setAnswer]         = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [city, setCity]             = useState('Mumbai')
  const [time, setTime]             = useState(new Date())

  // ── LIVE CLOCK ─────────────────────────────────────
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  // ── FETCH ALL DATA ─────────────────────────────────
  // runs when component first loads
  // fetches all endpoints from FastAPI
  useEffect(() => {
    fetchAllData()
  }, [city])

  async function fetchAllData() {
    setLoading(true)
    try {
      // fetch weather
      const w = await fetch(`${API}/api/predictions/weather?city=${city}`)
      setWeather(await w.json())

      // fetch ML prediction
      const p = await fetch(`${API}/api/predictions/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ city, model_name: 'random_forest' })
      })
      setPrediction(await p.json())

      // fetch sentiment
      const s = await fetch(`${API}/api/predictions/sentiment?city=${city}`)
      setSentiment(await s.json())

      // fetch anomaly
      const a = await fetch(`${API}/api/predictions/anomaly?city=${city}`)
      setAnomaly(await a.json())

      // fetch topics
      const t = await fetch(`${API}/api/predictions/topics?city=${city}`)
      setTopics(await t.json())

    } catch (err) {
      console.error('Failed to fetch data:', err)
    }
    setLoading(false)
  }

  // ── CHAT WITH GEMINI ───────────────────────────────
  async function askGemini() {
    if (!question.trim()) return
    setChatLoading(true)
    setAnswer('')
    try {
      const res = await fetch(`${API}/api/chat/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, city })
      })
      const data = await res.json()
      setAnswer(data.answer || data.detail || 'No answer received')
    } catch (err) {
      setAnswer('Error connecting to chatbot')
    }
    setChatLoading(false)
  }

  // ── RENDER ─────────────────────────────────────────
  return (
    <div style={styles.container}>

      {/* HEADER */}
      <div style={styles.header}>
        <div>
          <h1 style={styles.title}>🏙 SmartCity Pulse</h1>
          <p style={styles.subtitle}>Real-Time Urban Intelligence Platform</p>
        </div>
        <div style={styles.headerRight}>
          <div style={styles.timeBox}>
            {time.toLocaleTimeString()}
          </div>
          <select
            style={styles.citySelect}
            value={city}
            onChange={e => setCity(e.target.value)}
          >
            <option>Mumbai</option>
            <option>Delhi</option>
            <option>Pune</option>
            <option>Bangalore</option>
            <option>Chennai</option>
          </select>
          <button style={styles.refreshBtn} onClick={fetchAllData}>
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* LOADING */}
      {loading && (
        <div style={styles.loadingBox}>
          ⏳ Fetching live data for {city}...
        </div>
      )}

      {/* CARDS GRID */}
      {!loading && (
        <div style={styles.grid}>

          {/* WEATHER CARD */}
          {weather && (
            <div style={styles.card}>
              <h2 style={styles.cardTitle}>🌤 Current Weather</h2>
              <div style={styles.bigNumber}>{weather.temperature}°C</div>
              <p style={styles.cardText}>Feels like {weather.feels_like}°C</p>
              <p style={styles.cardText}>{weather.description}</p>
              <div style={styles.row}>
                <span style={styles.badge}>💧 {weather.humidity}%</span>
                <span style={styles.badge}>💨 {weather.wind_speed} m/s</span>
                <span style={styles.badge}>🌧 {weather.rain_1h}mm</span>
              </div>
              <div style={styles.forecastRow}>
                {weather.forecast && weather.forecast.map((f, i) => (
                  <div key={i} style={styles.forecastItem}>
                    <div style={styles.forecastTime}>{f.time}</div>
                    <div style={styles.forecastTemp}>{f.temperature}°</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ML PREDICTION CARD */}
          {prediction && (
            <div style={styles.card}>
              <h2 style={styles.cardTitle}>🤖 ML Prediction</h2>
              <div style={styles.bigNumber}>{prediction.prediction}</div>
              <p style={styles.cardText}>
                Confidence: {(prediction.confidence * 100).toFixed(0)}%
              </p>
              <div style={styles.confidenceBar}>
                <div style={{
                  ...styles.confidenceFill,
                  width: `${prediction.confidence * 100}%`
                }} />
              </div>
              <p style={styles.cardText}>Model: {prediction.model_used}</p>
              <p style={styles.cardText}>
                Temp: {prediction.temperature}°C |
                Humidity: {prediction.humidity}%
              </p>
            </div>
          )}

          {/* SENTIMENT CARD */}
          {sentiment && (
            <div style={styles.card}>
              <h2 style={styles.cardTitle}>📰 News Sentiment</h2>
              <div style={styles.row}>
                <div style={styles.sentimentBox('#27ae60')}>
                  <div style={styles.sentimentNum}>
                    {sentiment.positive_count}
                  </div>
                  <div>Positive</div>
                </div>
                <div style={styles.sentimentBox('#e74c3c')}>
                  <div style={styles.sentimentNum}>
                    {sentiment.negative_count}
                  </div>
                  <div>Negative</div>
                </div>
              </div>
              <div style={styles.sentimentBar}>
                <div style={{
                  width: `${sentiment.positive_pct}%`,
                  background: '#27ae60',
                  height: '100%',
                  borderRadius: '4px 0 0 4px'
                }} />
                <div style={{
                  width: `${100 - sentiment.positive_pct}%`,
                  background: '#e74c3c',
                  height: '100%',
                  borderRadius: '0 4px 4px 0'
                }} />
              </div>
              <p style={styles.cardText}>
                {sentiment.positive_pct}% positive today
              </p>
              <div style={styles.headlineList}>
                {sentiment.headlines.slice(0, 3).map((h, i) => (
                  <div key={i} style={styles.headlineItem}>
                    <span style={{
                      color: h.sentiment === 'POSITIVE' ? '#27ae60' : '#e74c3c',
                      marginRight: 6
                    }}>
                      {h.sentiment === 'POSITIVE' ? '✅' : '❌'}
                    </span>
                    {h.title.slice(0, 50)}...
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ANOMALY CARD */}
          {anomaly && (
            <div style={{
              ...styles.card,
              borderLeft: anomaly.anomaly_count > 0
                ? '4px solid #e74c3c'
                : '4px solid #27ae60'
            }}>
              <h2 style={styles.cardTitle}>⚠️ Anomaly Detection</h2>
              <div style={styles.bigNumber}>
                {anomaly.anomaly_count}
              </div>
              <p style={styles.cardText}>
                anomalies in {anomaly.total_records} records
              </p>
              {anomaly.anomalies.map((a, i) => (
                <div key={i} style={styles.anomalyItem}>
                  🔴 {a.time} — {a.temperature}°C,
                  wind {a.wind_speed} m/s
                </div>
              ))}
              {anomaly.anomaly_count === 0 && (
                <p style={{ color: '#27ae60' }}>
                  ✅ All weather readings normal
                </p>
              )}
            </div>
          )}

          {/* TOPICS CARD */}
          {topics && (
            <div style={styles.card}>
              <h2 style={styles.cardTitle}>📌 News Topics</h2>
              <p style={styles.cardText}>
                LDA found {topics.num_topics} topics in today's news
              </p>
              {topics.topics.map((t, i) => (
                <div key={i} style={styles.topicItem}>
                  <span style={styles.topicNum}>Topic {i + 1}</span>
                  <div style={styles.topicWords}>
                    {t.words.map((w, j) => (
                      <span key={j} style={styles.topicWord}>{w}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* CHATBOT CARD */}
          <div style={styles.card}>
            <h2 style={styles.cardTitle}>🤖 AI City Assistant</h2>
            <p style={styles.cardText}>
              Ask anything about {city} — powered by Gemini
            </p>
            <input
              style={styles.chatInput}
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && askGemini()}
              placeholder={`Ask about ${city}...`}
            />
            <button
              style={styles.chatBtn}
              onClick={askGemini}
              disabled={chatLoading}
            >
              {chatLoading ? '⏳ Thinking...' : '🚀 Ask Gemini'}
            </button>
            {answer && (
              <div style={styles.answerBox}>
                <strong>🤖 Answer:</strong>
                <p style={{ marginTop: 8 }}>{answer}</p>
              </div>
            )}
          </div>

        </div>
      )}

    </div>
  )
}

// ── STYLES ──────────────────────────────────────────
const styles = {
  container: {
    minHeight: '100vh',
    background: '#0f1117',
    color: '#e0e0e0',
    fontFamily: 'system-ui, sans-serif',
    padding: '0 0 40px 0'
  },
  header: {
    background: 'linear-gradient(135deg, #1a1f2e, #2d3561)',
    padding: '20px 32px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottom: '1px solid #2d3561'
  },
  title: {
    margin: 0,
    fontSize: 28,
    color: '#fff'
  },
  subtitle: {
    margin: '4px 0 0 0',
    color: '#8899aa',
    fontSize: 14
  },
  headerRight: {
    display: 'flex',
    gap: 12,
    alignItems: 'center'
  },
  timeBox: {
    background: '#1a1f2e',
    padding: '8px 16px',
    borderRadius: 8,
    fontSize: 18,
    fontWeight: 'bold',
    color: '#64b5f6'
  },
  citySelect: {
    background: '#1a1f2e',
    color: '#fff',
    border: '1px solid #2d3561',
    padding: '8px 12px',
    borderRadius: 8,
    fontSize: 14,
    cursor: 'pointer'
  },
  refreshBtn: {
    background: '#2d3561',
    color: '#fff',
    border: 'none',
    padding: '8px 16px',
    borderRadius: 8,
    cursor: 'pointer',
    fontSize: 14
  },
  loadingBox: {
    textAlign: 'center',
    padding: 60,
    fontSize: 18,
    color: '#8899aa'
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
    gap: 20,
    padding: 24
  },
  card: {
    background: '#1a1f2e',
    borderRadius: 12,
    padding: 24,
    border: '1px solid #2d3561'
  },
  cardTitle: {
    margin: '0 0 16px 0',
    fontSize: 16,
    color: '#8899aa',
    textTransform: 'uppercase',
    letterSpacing: 1
  },
  cardText: {
    color: '#8899aa',
    fontSize: 14,
    margin: '8px 0'
  },
  bigNumber: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#64b5f6',
    margin: '8px 0'
  },
  row: {
    display: 'flex',
    gap: 8,
    flexWrap: 'wrap',
    margin: '12px 0'
  },
  badge: {
    background: '#2d3561',
    padding: '4px 10px',
    borderRadius: 20,
    fontSize: 13
  },
  forecastRow: {
    display: 'flex',
    gap: 8,
    marginTop: 16
  },
  forecastItem: {
    background: '#2d3561',
    borderRadius: 8,
    padding: '8px 12px',
    textAlign: 'center',
    flex: 1
  },
  forecastTime: {
    fontSize: 11,
    color: '#8899aa'
  },
  forecastTemp: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#64b5f6'
  },
  confidenceBar: {
    background: '#2d3561',
    borderRadius: 4,
    height: 8,
    margin: '8px 0'
  },
  confidenceFill: {
    background: '#64b5f6',
    height: '100%',
    borderRadius: 4,
    transition: 'width 0.5s ease'
  },
  sentimentBox: (color) => ({
    flex: 1,
    background: color + '22',
    border: `1px solid ${color}`,
    borderRadius: 8,
    padding: 12,
    textAlign: 'center',
    color
  }),
  sentimentNum: {
    fontSize: 32,
    fontWeight: 'bold'
  },
  sentimentBar: {
    display: 'flex',
    height: 12,
    borderRadius: 4,
    overflow: 'hidden',
    margin: '12px 0'
  },
  headlineList: {
    marginTop: 12
  },
  headlineItem: {
    fontSize: 12,
    color: '#8899aa',
    padding: '4px 0',
    borderBottom: '1px solid #2d3561'
  },
  anomalyItem: {
    background: '#2d151522',
    border: '1px solid #e74c3c44',
    borderRadius: 6,
    padding: '8px 12px',
    fontSize: 13,
    margin: '6px 0',
    color: '#ff8a80'
  },
  topicItem: {
    margin: '10px 0'
  },
  topicNum: {
    fontSize: 12,
    color: '#8899aa',
    display: 'block',
    marginBottom: 4
  },
  topicWords: {
    display: 'flex',
    gap: 6,
    flexWrap: 'wrap'
  },
  topicWord: {
    background: '#2d3561',
    padding: '3px 10px',
    borderRadius: 20,
    fontSize: 12
  },
  chatInput: {
    width: '100%',
    background: '#0f1117',
    border: '1px solid #2d3561',
    color: '#fff',
    padding: '10px 12px',
    borderRadius: 8,
    fontSize: 14,
    marginBottom: 10,
    boxSizing: 'border-box'
  },
  chatBtn: {
    width: '100%',
    background: '#2d3561',
    color: '#fff',
    border: 'none',
    padding: '10px',
    borderRadius: 8,
    cursor: 'pointer',
    fontSize: 15,
    fontWeight: 'bold'
  },
  answerBox: {
    background: '#0f1117',
    border: '1px solid #2d3561',
    borderRadius: 8,
    padding: 12,
    marginTop: 12,
    fontSize: 14,
    lineHeight: 1.6
  }
}