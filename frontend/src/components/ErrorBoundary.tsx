import { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error,
      errorInfo: null
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error to console and diagnostics
    console.error('ErrorBoundary caught an error:', error, errorInfo);

    // Update state with error info
    this.setState({
      error,
      errorInfo
    });

    // Log to global diagnostics if available
    if ((window as any).__diagnostics) {
      (window as any).__diagnostics.logError('React Error Boundary caught error', error);
    }
  }

  private copyLogs = () => {
    try {
      // Get console logs from diagnostics overlay
      let logs = '';
      if ((window as any).__diagnostics) {
        logs = (window as any).__diagnostics.getLogs();
      }

      // Add error boundary specific info
      const errorLogs = `
=== ERROR BOUNDARY LOGS ===
Error: ${this.state.error?.message || 'Unknown error'}
Stack: ${this.state.error?.stack || 'No stack trace'}
Component Stack: ${this.state.errorInfo?.componentStack || 'No component stack'}

=== CONSOLE LOGS ===
${logs}

=== BROWSER INFO ===
User Agent: ${navigator.userAgent}
URL: ${window.location.href}
Timestamp: ${new Date().toISOString()}
      `.trim();

      // Copy to clipboard
      navigator.clipboard.writeText(errorLogs).then(() => {
        alert('Logs copied to clipboard!');
      }).catch(() => {
        // Fallback: show in prompt
        prompt('Copy these logs:', errorLogs);
      });
    } catch (err) {
      console.error('Failed to copy logs:', err);
      alert('Failed to copy logs. Check console for details.');
    }
  };

  private reloadPage = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: '#1a1a1a',
          color: '#ffffff',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: 'monospace',
          zIndex: 999999
        }}>
          <div style={{
            maxWidth: '600px',
            padding: '20px',
            backgroundColor: '#2a2a2a',
            border: '2px solid #ff4444',
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <h1 style={{ color: '#ff4444', marginBottom: '20px' }}>
              🚨 React App Crashed
            </h1>

            <div style={{
              backgroundColor: '#1a1a1a',
              padding: '15px',
              borderRadius: '4px',
              marginBottom: '20px',
              textAlign: 'left',
              fontSize: '14px'
            }}>
              <div style={{ color: '#ff6666', marginBottom: '10px' }}>
                <strong>Error:</strong> {this.state.error?.message || 'Unknown error'}
              </div>

              {this.state.error?.stack && (
                <div style={{ color: '#888', fontSize: '12px', whiteSpace: 'pre-wrap' }}>
                  <strong>Stack Trace:</strong>
                  {this.state.error.stack}
                </div>
              )}

              {this.state.errorInfo?.componentStack && (
                <div style={{ color: '#888', fontSize: '12px', whiteSpace: 'pre-wrap', marginTop: '10px' }}>
                  <strong>Component Stack:</strong>
                  {this.state.errorInfo.componentStack}
                </div>
              )}
            </div>

            <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
              <button
                onClick={this.copyLogs}
                style={{
                  backgroundColor: '#4CAF50',
                  color: 'white',
                  border: 'none',
                  padding: '10px 20px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '14px'
                }}
              >
                📋 Copy Logs
              </button>

              <button
                onClick={this.reloadPage}
                style={{
                  backgroundColor: '#2196F3',
                  color: 'white',
                  border: 'none',
                  padding: '10px 20px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '14px'
                }}
              >
                🔄 Reload Page
              </button>
            </div>

            <div style={{
              marginTop: '20px',
              fontSize: '12px',
              color: '#888',
              textAlign: 'left'
            }}>
              <div>If this error persists:</div>
              <div>1. Check the browser console for more details</div>
              <div>2. Try refreshing the page</div>
              <div>3. Check if the backend server is running</div>
              <div>4. Look for the red debug button (🐛) in the top-right corner</div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
