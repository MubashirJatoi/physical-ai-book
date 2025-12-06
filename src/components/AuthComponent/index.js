import React, { useState, useContext, createContext } from 'react';
import './AuthComponent.css';

// Create Auth Context
const AuthContext = createContext();

// Auth Provider Component
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);

  const login = async (email, password) => {
    setLoading(true);
    try {
      // In a real implementation, this would call an authentication API
      // For demo purposes, we'll simulate a successful login
      await new Promise(resolve => setTimeout(resolve, 1000));

      const mockUser = {
        id: 'demo-user-123',
        email,
        name: email.split('@')[0],
        preferences: {
          language: 'en',
          theme: 'light',
          notifications: true
        }
      };

      setUser(mockUser);
      localStorage.setItem('authUser', JSON.stringify(mockUser));
      return { success: true, user: mockUser };
    } catch (error) {
      return { success: false, error: error.message };
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('authUser');
  };

  const register = async (name, email, password) => {
    setLoading(true);
    try {
      // In a real implementation, this would call a registration API
      await new Promise(resolve => setTimeout(resolve, 1000));

      const mockUser = {
        id: `demo-user-${Date.now()}`,
        email,
        name,
        preferences: {
          language: 'en',
          theme: 'light',
          notifications: true
        }
      };

      setUser(mockUser);
      localStorage.setItem('authUser', JSON.stringify(mockUser));
      return { success: true, user: mockUser };
    } catch (error) {
      return { success: false, error: error.message };
    } finally {
      setLoading(false);
    }
  };

  // Check for existing user on initial load
  React.useEffect(() => {
    const storedUser = localStorage.getItem('authUser');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  return (
    <AuthContext.Provider value={{ user, login, logout, register, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Main AuthComponent that shows login/register forms
const AuthComponent = ({ mode = 'login' }) => {
  const { user, login, register, logout, loading } = useAuth();
  const [currentMode, setCurrentMode] = useState(mode);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: ''
  });

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    const result = await login(formData.email, formData.password);
    if (!result.success) {
      alert(`Login failed: ${result.error}`);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    const result = await register(formData.name, formData.email, formData.password);
    if (!result.success) {
      alert(`Registration failed: ${result.error}`);
    }
  };

  const toggleMode = () => {
    setCurrentMode(currentMode === 'login' ? 'register' : 'login');
  };

  if (user) {
    return (
      <div className="auth-component">
        <div className="auth-logged-in">
          <div className="user-info">
            <h4>Welcome, {user.name}!</h4>
            <p>{user.email}</p>
          </div>
          <button className="logout-btn" onClick={logout}>
            Logout
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-component">
      <div className="auth-form">
        <h3>{currentMode === 'login' ? 'Login' : 'Register'}</h3>

        <form onSubmit={currentMode === 'login' ? handleLogin : handleRegister}>
          {currentMode === 'register' && (
            <div className="form-group">
              <label htmlFor="name">Full Name</label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                required
              />
            </div>
          )}

          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleInputChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleInputChange}
              required
            />
          </div>

          <button
            type="submit"
            className="submit-btn"
            disabled={loading}
          >
            {loading ? 'Processing...' : currentMode === 'login' ? 'Login' : 'Register'}
          </button>
        </form>

        <div className="auth-toggle">
          <p>
            {currentMode === 'login'
              ? "Don't have an account? "
              : "Already have an account? "}
            <button className="toggle-btn" onClick={toggleMode}>
              {currentMode === 'login' ? 'Register' : 'Login'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default AuthComponent;