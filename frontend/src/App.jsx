import React, { useState } from 'react'
import Landing from './components/Landing'
import DomainSelection from './components/DomainSelection'
import Consultation from './components/Consultation'

function App() {
  const [view, setView] = useState('landing') // landing, domains, consultation
  const [selectedDomain, setSelectedDomain] = useState(null)

  const navigateToDomains = () => setView('domains')
  const navigateToLanding = () => setView('landing')
  const startConsultation = (domain) => {
    setSelectedDomain(domain)
    setView('consultation')
  }

  return (
    <div className="relative z-10 min-h-screen">
      {view === 'landing' && <Landing onGetConsulted={navigateToDomains} />}
      {view === 'domains' && (
        <DomainSelection 
          onBack={navigateToLanding} 
          onSelectDomain={startConsultation} 
        />
      )}
      {view === 'consultation' && (
        <Consultation 
          domain={selectedDomain} 
          onBack={navigateToDomains} 
        />
      )}
    </div>
  )
}

export default App
