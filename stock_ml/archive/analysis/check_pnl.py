"""Check if displayed PnL matches actual price changes on chart."""
import json

for sym in ['ACB', 'FPT', 'HPG', 'MBB', 'VNM', 'AAA', 'SSI', 'VND']:
    d = json.load(open(f'visualization/data/{sym}.json'))
    ohlcv = {c['time']: c for c in d['ohlcv']}
    markers = d['markers']
    
    print(f'\n{"="*80}')
    print(f'  {sym} — {len(markers)//2} trades')
    print(f'{"="*80}')
    
    mismatches = 0
    for i in range(0, len(markers), 2):
        if i + 1 >= len(markers):
            break
        buy = markers[i]
        sell = markers[i + 1]
        
        bp = ohlcv.get(buy['time'], {}).get('close', 0)
        sp = ohlcv.get(sell['time'], {}).get('close', 0)
        
        if bp <= 0:
            continue
        
        real_pnl = (sp - bp) / bp * 100
        shown_text = sell['text']
        
        # Extract shown PnL from text like "Exit +1.8% (5d)"
        import re
        match = re.search(r'([+-]?\d+\.?\d*)%', shown_text)
        shown_pnl = float(match.group(1)) if match else None
        
        if shown_pnl is not None:
            diff = abs(real_pnl - shown_pnl)
            flag = "❌ MISMATCH" if diff > 3.0 else ("⚠️ off" if diff > 1.0 else "✅")
            if diff > 1.0:
                mismatches += 1
            print(f'  {buy["time"]} @{bp:.2f} → {sell["time"]} @{sp:.2f} | Real: {real_pnl:+.1f}% | Shown: {shown_pnl:+.1f}% | Δ={diff:.1f}% {flag}')
    
    if mismatches == 0:
        print(f'  All trades match ✅')
    else:
        print(f'  ⚠️ {mismatches} mismatches found!')
