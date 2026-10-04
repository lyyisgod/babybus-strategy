"""Single account-level outlet; old modules remain independently callable."""
from collections import Counter
from copy import deepcopy

from .decision import Decision
from .closest import closest_candidate
from .inputs import (EvidenceError, finite, prices, price_metrics, member,
                     eps_revision, calendar_state, known)
from .macro import macro_facts
from .policy import official_policy


MAX_NAMES=16
MAX_INDUSTRY_NAMES=5
SATELLITE_NOMINAL_CAP=.02
COVERAGE_KEYS=('membership','consensus','five_day_gain','not_down','industry_cap',
               'new_high','deleveraging','metal_clock')
METALS={'SLV','SILJ','GLD','IAU','GOLD','SILVER'}


def metal_clock(regime,cut_index,expected_hikes,asof):
    if regime in {'HIKING','PAUSE'}:
        if (isinstance(expected_hikes,dict) and known(expected_hikes,asof)
                and isinstance(expected_hikes.get('value'),int)
                and not isinstance(expected_hikes['value'],bool) and expected_hikes['value']>0):
            return 'DIP_ALLOWED'
        return 'CLOSED_MISSING_VERIFIED_HIKES'
    if regime=='FIRST_CUT' or cut_index==1:
        return 'HOLD_FIRST_CUT'
    if cut_index==2:
        return 'EXIT_SECOND_CUT'
    if cut_index is not None and cut_index>=3:
        return 'OBSERVE_THIRD_PLUS'
    return 'CLOSED_UNKNOWN_POLICY'


def _book(book,asof):
    """NAV and nominal use the same currency; budget is available add capital."""
    positions=deepcopy([p for p in book.get('positions',[]) if not p.get('opened_on') or p['opened_on']<=asof])
    watches=deepcopy([p for p in book.get('watchlist',[]) if p.get('declared_on',asof)<=asof])
    nav,budget=book.get('nav'),book.get('budget')
    reasons=[]
    if not finite(nav) or nav<=0:
        reasons.append('missing_or_invalid_nav')
    if not finite(budget) or budget<=0:
        reasons.append('missing_or_invalid_budget')
    gross_known=finite(nav) and nav>0
    symbols=set()
    for p in positions:
        if not finite(p.get('nominal')):
            gross_known=False; reasons.append('missing_or_invalid_nominal')
        if p.get('symbol') in symbols:
            reasons.append('duplicate_book_symbol')
        symbols.add(p.get('symbol'))
        if finite(p.get('nominal')) and p['nominal']!=0 and not p.get('opened_on'):
            reasons.append('missing_opened_on')
    gross=sum(abs(p['nominal']) for p in positions)/nav if gross_known else None
    by_symbol={p.get('symbol'):p for p in positions}
    for p in watches:
        if p.get('symbol') not in by_symbol:
            by_symbol[p.get('symbol')]=p
    assets=sorted(by_symbol.values(),key=lambda p:(p.get('market',''),p.get('symbol','')))
    core=[p for p in assets if p.get('role')=='CORE']
    if len(core)>1:
        reasons.append('multiple_global_cores')
    promotion=book.get('core_promotion')
    if promotion:
        target=by_symbol.get(promotion.get('symbol'))
        old=by_symbol.get(promotion.get('previous_core'))
        if (promotion.get('issued_on','')>asof or not target or target.get('role')!='CORE'
                or old and old.get('role')=='CORE'):
            reasons.append('promotion_requires_old_core_exit_or_demotion')
    for p in core:
        if p.get('previous_role') in {'SATELLITE','BALLAST'} and (
            not promotion or promotion.get('symbol')!=p.get('symbol')):
            reasons.append('explicit_core_promotion_required')
    total_limit=book.get('max_names',MAX_NAMES)
    if not isinstance(total_limit,int) or isinstance(total_limit,bool) or not 1<=total_limit<=MAX_NAMES:
        reasons.append('invalid_total_name_cap'); total_limit=MAX_NAMES
    held=[p for p in positions if finite(p.get('nominal')) and p['nominal']!=0]
    count=Counter(p.get('industry') for p in held)
    return {'assets':assets,'cores':core,'held':held,'gross':gross,'budget_ready':not reasons,
            'reasons':sorted(set(reasons)),'industry_counts':count,'names':len(held),'max_names':total_limit}


def decide(book,facts,policy_statements,asof):
    """Return one frozen Decision from raw facts, a ledger and official history.

    Surge estimation is intentionally outside this function. Attach it using
    Decision.with_surge; no estimate is read by any account-level rule.
    """
    policy=official_policy(policy_statements,asof)
    macro=macro_facts(facts,asof)
    account=_book(book,asof)
    pressure=macro['pressure']
    regime=policy['regime']
    clock=metal_clock(regime,policy['cut_index'],facts.get('expected_hikes_remaining'),asof)
    coverage={k:0 for k in COVERAGE_KEYS}
    constraints={'allow_margin':not pressure and regime=='HIKING',
                 'max_gross':1. if pressure else None,'max_cores':1,
                 'max_industry_names':MAX_INDUSTRY_NAMES,'max_names':account['max_names'],
                 'satellite_suggested_nominal_cap':SATELLITE_NOMINAL_CAP,
                 'allow_new_core':regime in {'HIKING','PAUSE'},
                 'allow_new_satellite':regime in {'HIKING','PAUSE'} and macro['yield_stable'] is not True}
    if not account['budget_ready'] or not macro['data_valid']:
        constraints['allow_new_core']=constraints['allow_new_satellite']=False
    cores,satellites,rejects=[],[],[]
    computed={}
    contexts={}
    calendars={m:calendar_state(facts,m,asof) for m in {'US','CN'}}
    names=account['names']; industries=account['industry_counts'].copy()
    held_symbols={p['symbol'] for p in account['held']}
    def reject(p,reason,count=None):
        rejects.append({'symbol':p.get('symbol'),'market':p.get('market'),'reason':reason})
        if count:coverage[count]+=1
    for p in account['assets']:
        symbol,market,role=p.get('symbol'),p.get('market'),p.get('role')
        contexts[symbol]={'names':names,'industry_names':industries[p.get('industry')]}
        cal=calendars.get(market,{})
        if role not in {'CORE','SATELLITE','BALLAST'} or market not in {'US','CN'} or not symbol or not p.get('industry'):
            reject(p,'unsupported_book_asset');continue
        if market=='CN' and (role!='BALLAST' or p['industry']!='SOE_DIVIDEND'):
            reject(p,'a_share_ballast_only','membership');continue
        if not cal.get('verified'):
            reject(p,'exchange_calendar_unverified');continue
        if not cal['open']:
            reject(p,'exchange_closed_next_open='+str(cal['next_open']));continue
        if market=='CN' and not member(facts,'CSI300',symbol,asof,'CN'):
            reject(p,'not_current_csi300_member','membership');continue
        if role=='CORE':
            if len(account['cores'])!=1:
                reject(p,'multiple_global_cores');continue
            if market!='US' or not member(facts,'SMH',symbol,asof):
                reject(p,'not_current_smh_member','membership');continue
        elif symbol not in {'SLV','SILJ','TLT'} and p['industry'] not in {'UTILITY','SOE_DIVIDEND'}:
            reject(p,'not_rate_suppressed_satellite_bucket');continue
        try:
            s=prices(facts,symbol,asof,market)
            metrics=price_metrics(s);computed[symbol]=metrics
        except EvidenceError as error:
            reject(p,str(error));continue
        if role=='CORE':
            try:
                consensus=eps_revision(facts,symbol,metrics['window_start'],asof)
            except EvidenceError as error:
                reject(p,str(error),'consensus');continue
            row={'symbol':symbol,'market':market,'role':role,'industry':p['industry'],
                 'status':'观察','metrics':metrics,'consensus':consensus}
            cores.append(row)
            if metrics['ret_5d']>.08:
                row['reason']='five_day_gain_gt_8pct';coverage['five_day_gain']+=1
            elif metrics['ret_1d']>=0 or metrics['down_candle'] is False:
                row['reason']='core_not_down';coverage['not_down']+=1
            elif metrics['down_candle'] is None:
                row['reason']='candle_open_missing'
            else:
                earnings=facts.get('earnings',{}).get(symbol)
                earnings_date=earnings.get('date') if isinstance(earnings,dict) else earnings
                if earnings_date and asof<earnings_date:
                    row['reason']='before_earnings_hold_only'
                elif metrics['dip_unit']==0:
                    row['reason']='below_frozen_dip_tier'
                elif symbol not in held_symbols and industries[p['industry']]>=MAX_INDUSTRY_NAMES:
                    row['reason']='sixth_industry_name';coverage['industry_cap']+=1
                elif symbol not in held_symbols and names>=account['max_names']:
                    row['reason']='total_name_cap'
                else:
                    row['reason']='closed_dip_with_same_fiscal_consensus'
                    row['status']='候选'
                    if symbol not in held_symbols:
                        names+=1;industries[p['industry']]+=1
            continue
        metal=symbol in METALS
        # A separate clock, not a stock HOLD or an equity FIRST_CUT exit.
        if metal and clock!='DIP_ALLOWED':
            reject(p,'metal_clock_'+clock,'metal_clock');continue
        if macro['yield_stable'] is True and symbol not in held_symbols:
            reject(p,'yield_stable_no_new_satellites');continue
        if regime not in {'HIKING','PAUSE'}:
            reject(p,'policy_no_new_satellites');continue
        if metrics['at_252_high'] is None:
            reject(p,'prior_252_price_history_missing');continue
        if symbol not in held_symbols and metrics['at_252_high']:
            reject(p,'new_satellite_at_252_high','new_high');continue
        if not ((metrics['rsi14'] is not None and metrics['rsi14']<30) or metrics['new_low20'] is True):
            reject(p,'satellite_not_oversold');continue
        if symbol not in held_symbols:
            if industries[p['industry']]>=MAX_INDUSTRY_NAMES:
                reject(p,'sixth_industry_name','industry_cap');continue
            if names>=account['max_names']:
                reject(p,'total_name_cap');continue
            names+=1;industries[p['industry']]+=1
        satellites.append({'symbol':symbol,'market':market,'role':role,'industry':p['industry'],
                           'status':'观察','metrics':metrics,'suggested_nominal_cap':.02})
    held_metals=sorted(p['symbol'] for p in account['held'] if p['symbol'] in METALS)
    core_held=any(p.get('symbol') in held_symbols for p in account['cores'])
    action,size='OBSERVE',0.
    reason=[]
    # Account risk is first: never issue a second add/exit on this same day.
    if pressure and account['gross'] is not None and account['gross']>1:
        action='DELEVER_TO_1X';reason=['confirmed_pressure_gross_gt_1']
        coverage['deleveraging']=len(account['assets'])
        cores=[];satellites=[]
    elif not macro['data_valid'] or not calendars['US']['open']:
        reason=['required_macro_evidence_or_us_session_missing']
    elif not account['budget_ready']:
        reason=account['reasons']
    elif regime=='UNKNOWN':
        reason=['official_regime_unknown']
    elif regime=='FIRST_CUT':
        action='CORE_EXIT_WINDOW' if core_held else 'PRECIOUS_METALS_HOLD' if held_metals else 'HOLD'
        reason=['first_cut_equity_exit_window_metals_hold']
    elif clock=='EXIT_SECOND_CUT' and held_metals:
        action='PRECIOUS_METALS_EXIT';reason=['second_cut_metals_exit']
    elif regime=='EASING':
        if held_metals and clock=='OBSERVE_THIRD_PLUS':
            action='PRECIOUS_METALS_OBSERVE'
        elif held_metals and clock=='HOLD_FIRST_CUT':
            action='PRECIOUS_METALS_HOLD'
        else:
            action='HOLD'
        reason=['easing_no_new_equity_core']
    else:
        eligible=[p for p in cores if p['status']=='候选']
        if eligible:
            action='CORE_ADD_NO_MARGIN' if pressure or regime=='PAUSE' else 'CORE_ADD'
            size=eligible[0]['metrics']['dip_unit'];eligible[0]['status']='可执行'
            reason=['ledger_core_frozen_dip_ordinal']
        elif satellites:
            action='SATELLITE_DIP';reason=['verified_rate_suppressed_oversold_candidates']
            budget_fraction=book['budget']/book['nav']
            room=max(0.,1-account['gross']) if pressure else budget_fraction
            for p in satellites:
                position=next((x for x in account['held'] if x['symbol']==p['symbol']),{})
                existing=abs(position.get('nominal',0))/book['nav']
                cap=max(0.,min(.02-existing,budget_fraction,room))
                p['suggested_nominal_cap']=cap
                p['status']='可执行' if cap>0 else '观察'
                budget_fraction-=cap;room-=cap
            if not any(p['status']=='可执行' for p in satellites):
                action='HOLD';reason=['satellite_nominal_or_gross_cap_no_room']
        else:
            action='CORE_HOLD_NO_MARGIN' if pressure and core_held else 'HOLD'
            reason=['no_eligible_add']
    # All other targets stay observations; an account has exactly one action.
    if action not in {'CORE_ADD','CORE_ADD_NO_MARGIN'}:
        for p in cores:p['status']='观察'
    if action!='SATELLITE_DIP':
        for p in satellites:p['status']='观察'
    trace={'asof':asof,'policy':policy,'macro':macro['trace'],'gross':account['gross'],
           'budget_ready':account['budget_ready'],'book_reasons':account['reasons'],
           'execution_status':'观察' if not account['budget_ready'] or not macro['data_valid']
                              else '证据齐备',
           'coverage':{'input_names':len(account['assets']),'cores':len(cores),
                       'satellites':len(satellites),'rejects':len(rejects),'blocked':coverage},
           'calendars':calendars,'metal_clock':clock,'metal_holdings':held_metals,
           'yield_stable':macro['yield_stable'],
           'reason':reason,'computed_price_facts':computed,
           'fedwatch_label':deepcopy(facts.get('fedwatch')) if isinstance(facts.get('fedwatch'),dict)
                            and known(facts['fedwatch'],asof) else None,
           'paper_n':0,'size_kind':'ordinal_not_nav_fraction'}
    trace['closest']=closest_candidate(
        account,facts,asof,regime=regime,macro=macro,calendars=calendars,clock=clock,
        contexts=contexts,computed=computed,cores=cores,satellites=satellites,
        metal_symbols=METALS,max_industry_names=MAX_INDUSTRY_NAMES,
        satellite_cap=SATELLITE_NOMINAL_CAP,book=book)
    if trace['closest'] is None:
        trace['reason']=['empty_book']
    return Decision(regime,macro['stress'],constraints,action,tuple(cores),tuple(satellites),
                    tuple(rejects),trace,size)
