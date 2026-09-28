function JPRInput({ label, name, value, onChange, tooltip }) {
  return (
    <div className="field-card compact-field">
      <div className="field-header">
        <label htmlFor={name}>{label}</label>
        <span className="tooltip-wrap" tabIndex="0" aria-label={tooltip} title={tooltip}>
          i
        </span>
      </div>
      <input
        id={name}
        name={name}
        type="number"
        min="0"
        max="10"
        step="0.1"
        value={value}
        onChange={(event) => onChange(name, Number(event.target.value))}
        placeholder="0.0"
      />
    </div>
  )
}

export default JPRInput
