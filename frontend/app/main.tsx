import { createRoot } from 'react-dom/client';
import Editor from './page';
import DecoupledWbcPage from './decoupled-wbc-page';
import ControllerPage from './controller-page';
import './globals.css';
const path = window.location.pathname.replace(/\/+$/, '') || '/';
createRoot(document.getElementById('root')!).render(path === '/decoupled-wbc' ? <DecoupledWbcPage/> : path === '/controller' ? <ControllerPage/> : <Editor />);
