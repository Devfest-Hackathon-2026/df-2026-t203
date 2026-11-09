import { Route, Routes } from 'react-router-dom'
import NavBar from './components/NavBar'
import Home from './pages/Home'
import Training from './pages/Training'
import Analysis from './pages/Analysis'

export default function App() {
  return (
    <div className="flex h-full flex-col">
      <NavBar />
      <main className="relative min-h-0 flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/training" element={<Training />} />
          <Route path="/analysis" element={<Analysis />} />
        </Routes>
      </main>
    </div>
  )
}
