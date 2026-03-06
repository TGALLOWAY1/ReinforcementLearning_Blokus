import { useEffect } from 'react';

const CRITICAL_ELEMENT_IDS = [
    'critical-ui-hint',
    'critical-ui-pass',
    'critical-ui-save',
    'critical-ui-settings'
];

/**
 * UIPersistenceGuard is a non-rendering component that monitors 
 * specific "critical" UI elements identified by data-testid.
 * If these elements are missing from the DOM while this component is mounted,
 * it logs a warning. This helps catch regression bugs where parts of the
 * header or controls disappear during specific game states (like AI turns).
 */
export const UIPersistenceGuard: React.FC = () => {
    useEffect(() => {
        // Only run in development or when explicitly enabled
        if (!import.meta.env.DEV) return;

        const checkPersistence = () => {
            const missing = CRITICAL_ELEMENT_IDS.filter(id => {
                const element = document.querySelector(`[data-testid="${id}"]`);
                return !element;
            });

            if (missing.length > 0) {
                console.warn(
                    `[UIPersistenceGuard] Critical UI elements missing from DOM: ${missing.join(', ')}. ` +
                    `Check conditional rendering logic in Play.tsx.`
                );
            }
        };

        // Check periodically
        const intervalId = setInterval(checkPersistence, 2000);

        // Initial check after a short delay to allow for rendering
        const initialTimeout = setTimeout(checkPersistence, 500);

        return () => {
            clearInterval(intervalId);
            clearTimeout(initialTimeout);
        };
    }, []);

    return null;
};
