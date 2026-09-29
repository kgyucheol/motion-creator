import { useState, type ChangeEvent, type InputHTMLAttributes } from 'react';
import { isCompleteNumericText } from '../lib/numeric-input';

type Props = Omit<InputHTMLAttributes<HTMLInputElement>, 'value' | 'defaultValue' | 'type'> & {
  value?: number | string;
  defaultValue?: number | string;
};

/** Keep an unfinished number visible while the parent stores only valid numeric values. */
export default function NumericInput({ value, defaultValue, className, onChange, onBlur, onFocus, onKeyDown, ...props }: Props) {
  const [draft, setDraft] = useState<string | null>(null);
  const [committed, setCommitted] = useState(String(defaultValue ?? ''));
  const displayed = value === undefined ? committed : String(value);
  const change = (event: ChangeEvent<HTMLInputElement>) => {
    const text = event.target.value;
    setDraft(text);
    if (isCompleteNumericText(text)) {
      if (value === undefined) setCommitted(text);
      onChange?.(event);
    }
  };
  return <input {...props} className={`numeric-input ${className ?? ''}`} type="text" inputMode="decimal" value={draft ?? displayed}
    onFocus={event => { setDraft(displayed); onFocus?.(event); }}
    onChange={change}
    onBlur={event => { setDraft(null); onBlur?.(event); }}
    onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur(); onKeyDown?.(event); }}/>
}
