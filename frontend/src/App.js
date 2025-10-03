import React, { useState } from "react";

function App() {
  const [features, setFeatures] = useState({
    area: "",
    bedrooms: "",
    bathrooms: "",
    floor: "",
    city: "",
    furnishing: "",
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const cities = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Pune", "Kolkata"];

  const handleChange = (e) => {
    setFeatures({ ...features, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    try {
     const response = await fetch("http://127.0.0.1:5000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ features }),
});


      if (!response.ok) throw new Error("Server error");

      const data = await response.json();
      if (data.predicted_price !== undefined) {
        setResult(data.predicted_price);
      } else if (data.error) {
        setError(data.error);
      }
    } catch (err) {
      setError("Failed to fetch data. Check if backend is running.");
    }
  };

  return (
    <div style={{ maxWidth: "500px", margin: "50px auto", fontFamily: "Arial, sans-serif" }}>
      <h1 style={{ textAlign: "center", color: "#2c3e50" }}>Indian House Price Predictor</h1>
      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "15px" }}>
        
        <input type="number" name="area" placeholder="Area (sqft)" value={features.area} onChange={handleChange} required />
        <input type="number" name="bedrooms" placeholder="Bedrooms" value={features.bedrooms} onChange={handleChange} required />
        <input type="number" name="bathrooms" placeholder="Bathrooms" value={features.bathrooms} onChange={handleChange} required />
        <input type="number" name="floor" placeholder="Floor" value={features.floor} onChange={handleChange} required />

        <select name="city" value={features.city} onChange={handleChange}>
          {cities.map((city) => (
            <option key={city} value={city}>{city}</option>
          ))}
        </select>

        <select name="furnishing" value={features.furnishing} onChange={handleChange}>
          <option value="Furnished">Furnished</option>
          <option value="Semi-Furnished">Semi-Furnished</option>
          <option value="Unfurnished">Unfurnished</option>
        </select>

        <button type="submit" style={{ padding: "10px", backgroundColor: "#3498db", color: "white", border: "none", cursor: "pointer" }}>
          Predict Price
        </button>
      </form>

      {result && <h3 style={{ marginTop: "20px", color: "green" }}>Predicted Price: ₹ {result}</h3>}
      {error && <p style={{ marginTop: "20px", color: "red" }}>{error}</p>}
    </div>
  );
}

export default App;
