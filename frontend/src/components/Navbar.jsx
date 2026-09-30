const menuItems = [
  { label: 'Home', href: '#home' },
  { label: 'Screening', href: '#screening' },
  { label: 'How It Works', href: '#how-it-works' },
  { label: 'AI Model', href: '#ai-model' },
  { label: 'About', href: '#about' },
]

function Navbar({ onStartScreening }) {
  return (
    <header className="topbar">
      <div className="container nav-shell">
        <a className="brand" href="#home" aria-label="OA-SCAN home">
          <span className="brand-mark">OA</span>
          <span className="brand-text">SCAN</span>
        </a>

        <nav className="main-nav" aria-label="Main navigation">
          {menuItems.map((item) => (
            <a key={item.label} href={item.href}>
              {item.label}
            </a>
          ))}
        </nav>

        <button type="button" className="primary-button nav-button" onClick={onStartScreening}>
          Start Screening
        </button>
      </div>
    </header>
  )
}

export default Navbar
