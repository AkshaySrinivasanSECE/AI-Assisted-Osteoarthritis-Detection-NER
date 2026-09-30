function PainSlider({ value, onChange }) {
  return (
    <div className="pain-slider-panel">
      <div className="slider-header-row">
        <label htmlFor="vas-score">Knee Pain Level</label>
        <div className="slider-value" aria-live="polite">{value}</div>
      </div>

      <input
        id="vas-score"
        type="range"
        min="0"
        max="10"
        step="1"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />

      <div className="slider-scale">
        <span>0 = No Pain</span>
        <span>10 = Worst Pain</span>
      </div>
    </div>
  )
}

export default PainSlider
