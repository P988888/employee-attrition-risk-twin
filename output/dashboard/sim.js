/* 办公行为模拟引擎 —— 部门自主决策 + 寻路 + 数据驱动（单层版）
 *
 * 设计目标：让整个办公层「活起来」—— 每个员工按自己的日程走动、开会、工作、休息；
 * 部门有各自的节奏（站会/例会/复盘/面试），而不是所有人做同一件事。
 *
 * 单层平面没有电梯，所以不存在跨层通勤：所有人的动线都发生在同一层的走廊上，
 * 状态机里也就没有 elevator 阶段，elevator/stairs 只是场景装饰。
 *
 * 三类输入：
 *   1. 布局（building.json）  ：房间矩形、工位坐标、走廊节点图
 *   2. 档案（profiles.json）  ：风险等级、部门、行为指标（专注/迟到/压力）
 *   3. 部门议程（本文件定义）：每个部门一天里自己决定的会议与外出安排
 *
 * 输出：任意时刻 T，每个人的 { floor, x, y, state, facing, speed }
 *
 * 状态（state）：
 *   work     在工位做事（坐着，手臂有敲键盘的微动）
 *   walk     沿走廊行走
 *   meeting  在会议室开会（坐着）
 *   pantry   茶水间（站着）
 *   lounge   休息区（坐着）
 *   print    打印区
 *   away     外出 / 不在楼内
 */

export const WORK_START = 9 * 60;    // 09:00
export const WORK_END = 18 * 60;     // 18:00
export const LUNCH_START = 12 * 60;
export const LUNCH_END = 13 * 60;
export const LEAD_MIN = 1.6;      // 提前动身时间（分钟）

/* 部门议程：各部门「自己决定」的固定安排
 * kind: meeting(会议室开会) / away(外出) / atDesk(在工位集体活动)
 *
 * 房间 id 全部对应单层平面里的实际房间（见 src/step7_office_layout.py）：
 *   meeting_l 大会议室 / meeting_s 小会议室 / pantry 茶水间
 *   lounge 休息区 / cafe 咖啡吧 / print 打印储物 / focus 专注区
 */
export const DEPT_AGENDA = {
  service: [
    { start: 9*60+15, end: 9*60+30,  kind: 'atDesk',  label: '班前交接' },
    { start: 12*60,   end: 13*60,    kind: 'lunch',   label: '午休轮值' },
    { start: 17*60+30,end: 18*60,    kind: 'atDesk',  label: '日报汇总' },
  ],
  rnd: [
    { start: 10*60,   end: 10*60+15, kind: 'meeting', room: 'meeting_s', label: '每日站会' },
    { start: 12*60,   end: 13*60,    kind: 'lunch',   label: '午休' },
    { start: 15*60,   end: 16*60,    kind: 'meeting', room: 'meeting_l', label: '代码评审' },
  ],
  marketing: [
    { start: 11*60,   end: 12*60,    kind: 'meeting', room: 'meeting_l', label: '周例会' },
    { start: 12*60,   end: 13*60,    kind: 'lunch',   label: '午休' },
    { start: 14*60,   end: 16*60,    kind: 'away',    label: '客户拜访' },
  ],
  ops: [
    { start: 12*60,   end: 13*60,    kind: 'lunch',   label: '午休' },
    { start: 14*60,   end: 15*60,    kind: 'meeting', room: 'meeting_s', label: '运营复盘' },
  ],
  finance: [
    { start: 9*60+30, end: 10*60+30, kind: 'atDesk',  label: '对账' },
    { start: 12*60,   end: 13*60,    kind: 'lunch',   label: '午休' },
    { start: 16*60,   end: 17*60+30, kind: 'meeting', room: 'meeting_l', label: '月度结算' },
  ],
  hr: [
    { start: 10*60,   end: 11*60,    kind: 'meeting', room: 'meeting_s', label: '面试' },
    { start: 12*60,   end: 13*60,    kind: 'lunch',   label: '午休' },
    { start: 15*60,   end: 16*60,    kind: 'meeting', room: 'meeting_s', label: '员工谈话' },
  ],
  exec: [
    { start: 12*60,   end: 13*60,    kind: 'lunch',   label: '午休' },
    { start: 14*60,   end: 16*60,    kind: 'away',    label: '外部会晤' },
    { start: 16*60,   end: 17*60,    kind: 'meeting', room: 'meeting_l', label: '高管例会' },
  ],
};

/* ---------------------------------------------------------------- 图与寻路 */
export class FloorGraph {
  constructor(floor, nodes) {
    this.floor = floor;
    this.nodes = {};             // id -> {x,y,links:[]}
    nodes.forEach(n => this.nodes[n.id] = { x: n.x, y: n.y, links: n.links.slice() });
    // 门节点 ↔ 房间
    this.roomDoor = {};
    Object.keys(this.nodes).forEach(id => {
      if (id.startsWith('door_')) this.roomDoor[id.slice(5)] = id;
    });
  }
  /** 从任意房间取「接入点」节点 id */
  access(roomId) {
    if (this.roomDoor[roomId]) return this.roomDoor[roomId];
    // 单层没有电梯厅，兜底用走廊骨架的第一个节点
    const spine = Object.keys(this.nodes).find(id => id.startsWith('c'));
    return spine || Object.keys(this.nodes)[0];
  }
  /** BFS 最短路，返回节点 id 数组 */
  path(fromId, toId) {
    if (fromId === toId) return [fromId];
    if (!this.nodes[fromId] || !this.nodes[toId]) return [];
    const prev = { [fromId]: null }, q = [fromId];
    while (q.length) {
      const cur = q.shift();
      if (cur === toId) break;
      for (const nb of this.nodes[cur].links) {
        if (!(nb in prev)) { prev[nb] = cur; q.push(nb); }
      }
    }
    if (!(toId in prev)) return [];
    const out = [];
    for (let c = toId; c; c = prev[c]) out.unshift(c);
    return out;
  }
  pos(id) { const n = this.nodes[id]; return n ? { x: n.x, y: n.y } : null; }
}

/** 判断点在哪个房间里（用于算路径起点） */
function roomAtPoint(layout, floor, x, y) {
  const f = layout.floors.find(v => v.level === floor);
  if (!f) return null;
  for (const r of f.rooms) {
    if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.d) return r.rid;
  }
  return null;
}

function roomForActivity(layout, agent, block) {
  const rooms = layout.floors.flatMap(f => f.rooms.map(room => ({ floor: f.level, room })));
  if (block.state === 'meeting' && block.room) {
    return rooms.find(x => x.room.rid === block.room) || null;
  }
  const matches = x => {
    const rid = x.room.rid;
    if (block.state === 'pantry' || block.state === 'lunch')
      return rid === 'pantry' || rid === 'cafe';
    if (block.state === 'lounge') return rid === 'lounge' || rid === 'cafe';
    if (block.state === 'print') return rid === 'storage';
    return false;
  };
  // 单层：所有房间都在同一层，直接取第一个匹配的房间
  return rooms.find(matches) || null;
}

function activitySlots(room, state) {
  const margin = state === 'meeting' ? 1.0 : 0.8;
  const w = Math.max(1.2, room.w - margin * 2);
  const d = Math.max(1.2, room.d - margin * 2);
  const cols = Math.max(2, Math.floor(w / 1.05));
  const rows = Math.max(2, Math.floor(d / 1.05));
  const slots = [];
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      slots.push({
        x: room.x + margin + (col + 0.5) * (w / cols),
        y: room.y + margin + (row + 0.5) * (d / rows),
      });
    }
  }
  return slots;
}

/* ---------------------------------------------------------------- 日程生成 */
function rnd(seed) {                       // 确定性伪随机（保证可拖拽回放一致）
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

function hash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

/**
 * 为一位员工生成一天的日程块。
 * 行为节奏由岗位、部门、行为资料和员工编号种子驱动；离职风险只用于看板展示，
 * 不参与自主行为决策，避免把预测标签直接当成人员行为规则。
 */
export function buildDailyPlan(emp, layout) {
  const R = rnd(hash(emp.employee_id));
  const dept = emp.department_key;
  const profile = emp.profile || {};
  const behavior = emp.behavior || {};
  const rhythmSeed = hash(`${emp.employee_id}|${dept}|${profile.JobRole || ''}`);
  const rhythm = (rhythmSeed % 1000) / 1000;

  const blocks = [];

  // ---- 上班 / 迟到 ----
  const lateMin = Math.round((behavior.late_days ? Math.min(18, behavior.late_days) : 6) * (0.35 + rhythm * 0.65));
  const arrive = WORK_START + lateMin;
  blocks.push({ start: 0, end: arrive, state: 'away', label: '未到岗' });

  // ---- 下班时间：由岗位节奏和稳定种子决定，不读取离职风险 ----
  const leaveEarly = rhythm > 0.93 ? 8 + Math.floor(R() * 12) : 0;
  const depart = WORK_END - leaveEarly;

  // ---- 部门议程 ----
  const agenda = DEPT_AGENDA[dept] || [];

  // ---- 随机离座（茶水间 / 打印 / 休息）----
  const breaks = [];
  const nBreak = 3 + Math.floor(R() * 3 + Math.min(2, (behavior.avg_meetings || 0) / 3));
  for (let i = 0; i < nBreak; i++) {
    const s = arrive + 30 + Math.floor(R() * Math.max(10, depart - arrive - 90));
    const dur = 5 + Math.floor(R() * 10);
    const kind = R() < 0.5 ? 'pantry' : (R() < 0.62 ? 'print' : 'lounge');
    breaks.push({ start: s, end: s + dur, state: kind, label:
      kind === 'pantry' ? '茶水间' : kind === 'print' ? '打印' : '休息区' });
  }

  // ---- 去同事工位讨论（办公室里最常见的走动）----
  const talks2 = [];
  const nTalk = 3 + Math.floor(R() * 4);
  for (let i = 0; i < nTalk; i++) {
    const s = arrive + 40 + Math.floor(R() * Math.max(10, depart - arrive - 120));
    talks2.push({ start: s, end: s + 4 + Math.floor(R() * 8),
                  state: 'discuss', label: '去同事工位讨论' });
  }

  // ---- 巡场走动：去别的区域转一圈（走廊里就会有人）
  const strolls = [];
  const nStroll = 2 + Math.floor(R() * 3);
  for (let i = 0; i < nStroll; i++) {
    const st = arrive + 25 + Math.floor(R() * Math.max(10, depart - arrive - 80));
    strolls.push({ start: st, end: st + 3 + Math.floor(R() * 5),
                  state: 'stroll', label: '巡场走动' });
  }

  // 固定议程优先，随机活动只填充未被固定议程占用的时间段。
  // 这样午休、会议不会和随机休息/讨论重叠，也能保证重复生成结果一致。
  const fixed = agenda.map(e => ({ ...e, priority: 0 }));
  const randomEvents = [...breaks, ...talks2, ...strolls].map(e => ({ ...e, priority: 1 }));
  const accepted = [...fixed];
  for (const ev of randomEvents.sort((a, b) => a.start - b.start || a.end - b.end)) {
    if (ev.end <= arrive || ev.start >= depart || ev.end <= ev.start) continue;
    if (!accepted.some(x => x.start < ev.end && ev.start < x.end)) accepted.push(ev);
  }
  const events = accepted.sort((a, b) => a.start - b.start || a.priority - b.priority);

  let cursor = arrive;
  for (const ev of events) {
    if (ev.end <= arrive || ev.start >= depart) continue;
    const s = Math.max(ev.start, arrive), e = Math.min(ev.end, depart);
    if (e <= cursor) continue;
    const start = Math.max(s, cursor);
    if (start > cursor) blocks.push({ start: cursor, end: start, state: 'work', label: '工位工作' });
    blocks.push({
      start, end: e,
      state: ev.kind === 'meeting' ? 'meeting'
           : ev.kind === 'away' ? 'away'
           : ev.kind === 'lunch' ? 'lunch'
           : ev.kind === 'atDesk' ? 'work'
           : ev.state || 'work',
      room: ev.room || null,
      label: ev.label || '',
      dept: true,
    });
    cursor = e;
  }
  if (cursor < depart) blocks.push({ start: cursor, end: depart, state: 'work', label: '工位工作' });
  blocks.push({ start: depart, end: 24 * 60, state: 'away', label: '已下班' });

  return blocks.sort((a, b) => a.start - b.start);
}

/* ---------------------------------------------------------------- 模拟器 */
export class Sim {
  /**
   * @param {object} layout  building.json
   * @param {Array}  profiles employee profiles (含 risk / desk / floor / department_key)
   */
  constructor(layout, profiles) {
    this.layout = layout;
    this.floors = {};
    this.rooms = {};                    // floor -> {rid: room}
    layout.floors.forEach(f => {
      this.floors[f.level] = new FloorGraph(f.level, f.waypoints);
      this.rooms[f.level] = {};
      f.rooms.forEach(r => this.rooms[f.level][r.rid] = r);
    });
    this.agents = profiles.map(p => this._makeAgent(p, layout));
    this._assignActivitySlots();
    // 从 00:00 开始建立状态，首次 seek 与任意回放路径使用同一条固定 tick 序列。
    this.t = 0;                         // 当前时间（分钟）
  }

  _makeAgent(p, layout) {
    return {
      id: p.employee_id, name: p.name,
      dept: p.department_key, deptCn: p.department_cn,
      floor: p.floor,
      homeFloor: p.floor,
      deskRoom: p.desk_room,
      desk: { x: p.desk.x, y: p.desk.y, rot: p.desk.rot },
      risk: (p.risk && p.risk.score) || 0.35,
      riskLevel: p.risk && p.risk.level,
      lead: 0.5 + (hash(p.employee_id) % 1000) / 1000 * 1.4,   // 动身错峰系数
      speed: 1.05 + (hash(p.employee_id + 'v') % 100) / 100 * 0.42,  // 个人步速 m/s
      v: 0,                      // 当前速度（有加减速）
      phase: 0,                  // 步态相位（由实际位移累积，脚不打滑）
      turn: p.desk.rot || 0,     // 平滑转向用的当前朝向
      plan: buildDailyPlan(p, layout),
      // 运行时
      x: p.desk.x, y: p.desk.y, floorNow: p.floor,
      state: 'work', facing: p.desk.rot || 0,
      path: [], pathIdx: 0, target: null,
    };
  }

  _assignActivitySlots() {
    const buckets = new Map();
    for (const a of this.agents) {
      for (const block of a.plan) {
        if (!['meeting', 'pantry', 'lunch', 'lounge', 'print'].includes(block.state)) continue;
        const found = roomForActivity(this.layout, a, block);
        if (!found) continue;
        const key = `${found.floor}|${found.room.rid}|${block.start}|${block.end}|${block.state}`;
        let bucket = buckets.get(key);
        if (!bucket) {
          bucket = { room: found.room, floor: found.floor, slots: activitySlots(found.room, block.state), next: 0 };
          buckets.set(key, bucket);
        }
        const slot = bucket.slots[bucket.next % bucket.slots.length];
        block._activityFloor = bucket.floor;
        block._activityRoom = bucket.room.rid;
        block._activitySlot = { x: slot.x, y: slot.y };
        block._activitySlotIndex = bucket.next++;
      }
    }
  }

  _resetAgent(a) {
    a.x = a.desk.x; a.y = a.desk.y; a.floorNow = a.homeFloor;
    a.state = 'work'; a.facing = a.desk.rot || 0; a.turn = a.desk.rot || 0;
    a.v = 0; a.phase = 0; a.moving = false;
    a.path = []; a.pathIdx = 0; a.target = null;
    a._key = null;
  }

  /** 该时刻此人的日程块 */
  blockAt(a, t) {
    for (const b of a.plan) if (t >= b.start && t < b.end) return b;
    return a.plan[a.plan.length - 1];
  }

  /**
   * 带「提前动身」的日程块选择。
   *
   * 直接按块开始时间才出发的话，步行只占 20-30 秒，一天里几乎看不到人走动，
   * 而且开会会迟到。这里让需要移动的下一块**提前一段**生效，保证按时到场。
   *
   * 提前量按人错峰（0.5×~1.9×）：否则全部门同时出发、同时到达，
   * 走廊会「一阵一阵」地空，看起来像列队行进而不是真实办公场景。
   */
  blockAtWithLead(a, t) {
    const cur = this.blockAt(a, t);
    const i = a.plan.indexOf(cur);
    const nxt = a.plan[i + 1];
    if (nxt && nxt.state !== 'away' && nxt.state !== cur.state) {
      const lead = (nxt.state === 'work') ? 0 : a.lead * LEAD_MIN;
      if (t >= nxt.start - lead) return nxt;
    }
    return cur;
  }

  /** 根据日程块算出目的地 */
  _destination(a, blk) {
    const L = this.layout;
    if (blk.state === 'work') {
      return { floor: a.homeFloor, room: a.deskRoom, x: a.desk.x, y: a.desk.y,
               rot: a.desk.rot, seat: true };
    }
    if (blk.state === 'meeting' && blk.room) {
      const f = L.floors.find(x => x.level === blk._activityFloor);
      const r = f && f.rooms.find(x => x.rid === blk._activityRoom || x.rid === blk.room);
      if (r) {
        const slot = blk._activitySlot || { x: r.x + r.w / 2, y: r.y + r.d / 2 };
        return { floor: f.level, room: r.rid, x: slot.x, y: slot.y, seat: true };
      }
    }
    if (blk.state === 'pantry' || blk.state === 'lounge' || blk.state === 'print'
        || blk.state === 'lunch') {
      const f = L.floors.find(x => x.level === blk._activityFloor);
      const r = f && f.rooms.find(x => x.rid === blk._activityRoom);
      if (r) {
        const slot = blk._activitySlot || { x: r.x + r.w / 2, y: r.y + r.d / 2 };
        return { floor: f.level, room: r.rid, x: slot.x, y: slot.y, seat: false };
      }
    }
    if (blk.state === 'stroll') {
      // 在本层随机挑一个「非自己房间」去转一圈 —— 走廊里自然就有人
      const fl = L.floors.find(f => f.level === a.homeFloor);
      const cands = fl.rooms.filter(r => r.rid !== a.deskRoom);
      if (cands.length){
        const r = cands[Math.floor(a.phase * 131 + a.lead * 977) % cands.length];
        return { floor: fl.level, room: r.rid,
                 x: r.x + r.w * (0.25 + (a.lead % 0.5)),
                 y: r.y + r.d * 0.5, seat: false };
      }
    }
    if (blk.state === 'discuss') {
      // 去本层另一个部门房间「讨论」——办公室最常见的走动
      const fl = L.floors.find(f => f.level === a.homeFloor);
      const cands = fl.rooms.filter(r => r.kind === 'dept' && r.rid !== a.deskRoom);
      if (cands.length){
        const r = cands[Math.floor(a.lead * 997) % cands.length];
        return { floor: fl.level, room: r.rid, x: r.x + r.w * 0.45,
                 y: r.y + r.d * 0.5, seat: false };
      }
    }
    return null;   // away
  }

  /** 重算路径（目的地变化时调用） */
  _repath(a, dest) {
    if (!dest) { a.path = []; a.target = null; return; }
    const g = this.floors[a.floorNow];
    // 起点用「此刻所在房间」而不是工位房间 —— 人可能正在茶水间/会议室
    const here = roomAtPoint(this.layout, a.floorNow, a.x, a.y) || a.deskRoom;
    const from = g.access(here);
    const to = g.access(dest.room);
    const pts = [{ x: a.x, y: a.y }];
    // 同一房间内：直达，不要绕走廊再绕回来。
    // 否则会出现死锁：人已在工位上 → 剩余距离 0 → 速度 0 → 位置不动。
    if (here === dest.room) {
      a.path = [{ x: a.x, y: a.y }, { x: dest.x, y: dest.y }];
      a.pathIdx = 0; a.target = dest;
      return;
    }
    const ids = g.path(from, to);
    ids.forEach(id => { const p = g.pos(id); if (p) pts.push(p); });
    pts.push({ x: dest.x, y: dest.y });
    a.path = pts; a.pathIdx = 0; a.target = dest;
  }

  /** 推进到目标时间（可大步快进，内部按固定步长积分） */
  seek(tMinutes) {
    const target = Math.max(0, Math.min(24*60, tMinutes));
    if (target < this.t) {                       // 回退：重置后重跑
      this.agents.forEach(a => this._resetAgent(a));
      this.t = 0;
    }
    // 固定 tick 让从不同起点 seek 到同一时刻得到相同快照。
    const STEP = 0.05;
    while (this.t < target) {
      const remain = target - this.t;
      const dt = Math.min(STEP, remain);
      this._step(dt);
      this.t += dt;
    }
    return this.t;
  }

  _step(dtMin) {
    const dt = dtMin * 60;                       // 秒
    for (const a of this.agents) {
      const blk = this.blockAtWithLead(a, this.t);
      const want = blk.state;
      const dest = this._destination(a, blk);

      // 目的地变了 → 重算路径
      const key = dest ? `${dest.room}|${want}|${blk.label}` : 'away';
      if (a._key !== key) { a._key = key; this._repath(a, dest); }

      if (!dest) { a.state = 'away'; a.moving = false; a.v = 0; continue; }

      // 防御：路径异常（空 / 只有一个点）时直接重建，避免后面的取值崩溃
      if (!a.path || a.path.length < 2){
        a.path = [{ x: a.x, y: a.y }, { x: dest.x, y: dest.y }];
        a.pathIdx = 0;
      }

      // 先跳过「已经站在上面」的路径点。
      // 否则会出现死锁：人已在工位上 → 剩余距离 0 → 速度 0 → 移动量 0
      // → pathIdx 不前进 → 永远判不到「已到达」→ 状态卡在「行走中」却原地不动。
      while (a.pathIdx < a.path.length - 1 &&
             Math.hypot(a.path[a.pathIdx + 1].x - a.x,
                        a.path[a.pathIdx + 1].y - a.y) < 0.08){
        a.pathIdx++;
      }
      // 目标速度：距离当前阶段终点 1.6m 内线性减速，避免急停
      const tail = Math.hypot(a.path[a.path.length-1].x - a.x,
                              a.path[a.path.length-1].y - a.y);
      const wantV = a.speed * Math.min(1, tail / 1.6);
      a.v += (wantV - a.v) * Math.min(1, dt * 2.2);       // 加速/减速过渡
      let remain = a.v * dt;
      while (remain > 0 && a.pathIdx < a.path.length - 1) {
        const p1 = a.path[a.pathIdx + 1];
        const dx = p1.x - a.x, dy = p1.y - a.y;
        const d = Math.hypot(dx, dy);
        if (d < 0.05) { a.pathIdx++; continue; }
        const step = Math.min(remain, d);
        a.x += dx / d * step; a.y += dy / d * step;
        a.phase += (step / 0.78) * Math.PI;                // 步幅 0.78m → 脚步与位移同步
        // 平滑转身：不要瞬间改朝向
        const wantFacing = Math.atan2(dy, dx) - Math.PI / 2;
        let df = wantFacing - a.turn;
        while (df > Math.PI) df -= Math.PI * 2;
        while (df < -Math.PI) df += Math.PI * 2;
        a.turn += df * Math.min(1, dt * 7);
        a.facing = a.turn;
        remain -= step;
        if (step >= d - 1e-6) a.pathIdx++;
      }
      if (a.pathIdx >= a.path.length - 1) a.v = 0;

      // 判断是否到达
      const arrived = a.pathIdx >= a.path.length - 1 &&
                      Math.hypot(a.x - a.path[a.path.length - 1].x,
                                 a.y - a.path[a.path.length - 1].y) < 0.35;
      if (arrived) {
        a.state = want; a.moving = false;
        if (dest.rot !== undefined && want === 'work') a.facing = dest.rot;
      } else {
        a.state = 'walk';
        a.moving = true;
      }
    }
  }

  /** 预分配快照缓冲（避免每帧 new 42 个对象 → GC 抖动） */
  initSnapshotBuffer() {
    this._buf = this.agents.map(() => ({
      id:null, x:0, y:0, floor:1, state:'work', facing:0, moving:false,
      risk:0, dept:null, label:'', phase:0, v:0,
    }));
    return this._buf;
  }

  /** 写入预分配缓冲并返回（不产生新对象） */
  fillSnapshot() {
    const buf = this._buf || this.initSnapshotBuffer();
    for (let i = 0; i < this.agents.length; i++) {
      const a = this.agents[i], o = buf[i];
      o.id = a.id; o.x = a.x; o.y = a.y; o.floor = a.floorNow;
      o.state = a.state; o.facing = a.facing; o.moving = !!a.moving;
      o.risk = a.risk; o.dept = a.dept;
      o.label = this.blockAtWithLead(a, this.t).label;
      o.phase = a.phase; o.v = a.v;
    }
    return buf;
  }

  /** 导出某一时刻所有人的渲染数据 */
  snapshot() {
    return this.agents.map(a => ({
      id: a.id, x: a.x, y: a.y, floor: a.floorNow, state: a.state,
      facing: a.facing, moving: !!a.moving, risk: a.risk, dept: a.dept,
      label: this.blockAtWithLead(a, this.t).label,
    }));
  }

  /** 统计当前时刻各状态人数（用于看板） */
  stats() {
    const s = {};
    this.agents.forEach(a => {
      const k = a.state;
      s[k] = (s[k] || 0) + 1;
    });
    return s;
  }

  /**
   * 按房间统计此刻的实时占用，供仪表盘的 KPI 和空间占用图表使用。
   * 返回 { rooms: [{rid, cn, kind, dept, seats, occupancy, people, cap, ratio}],
   *        onSite, seats, utilization, meetingRooms, meetingCap, meetingUsed, crowded }
   */
  roomOccupancy() {
    const f = this.layout.floors[0];
    const byRoom = new Map();
    for (const a of this.agents) {
      if (a.state === 'away') continue;
      const rid = roomAtPoint(this.layout, a.floorNow, a.x, a.y);
      if (!rid) continue;
      byRoom.set(rid, (byRoom.get(rid) || 0) + 1);
    }
    const rooms = f.rooms.map(r => {
      const people = byRoom.get(r.rid) || 0;
      // 座位房间用真实席位数做分母；公共区用「按面积估算的舒适容量」
      const cap = r.seats > 0 ? r.seats : Math.max(4, Math.round(r.w * r.d / 5));
      return {
        rid: r.rid, cn: r.cn, kind: r.kind, dept: r.dept || null,
        seats: r.seats || 0, people, cap,
        ratio: cap > 0 ? people / cap : 0,
      };
    });
    const seatRooms = rooms.filter(r => r.seats > 0);
    const seats = seatRooms.reduce((n, r) => n + r.cap, 0);
    const seated = seatRooms.reduce((n, r) => n + r.people, 0);
    const meetingRooms = rooms.filter(r => r.kind === 'meeting');
    const meetingCap = meetingRooms.length;
    const meetingUsed = meetingRooms.filter(r => r.people > 0).length;
    const onSite = this.agents.filter(a => a.state !== 'away').length;
    return {
      rooms, onSite, seats, utilization: seats ? seated / seats : 0,
      meetingRooms, meetingCap, meetingUsed,
      meetingRate: meetingCap ? meetingUsed / meetingCap : 0,
      crowded: rooms.filter(r => r.ratio >= 0.8).length,
    };
  }

  /** 每个部门的实时人数与工位利用率（用于部门对比图） */
  deptStats() {
    const seen = new Map();
    for (const a of this.agents) {
      const d = seen.get(a.dept) || { key: a.dept, cn: a.deptCn, total: 0, onSite: 0, working: 0 };
      d.total++;
      if (a.state !== 'away') d.onSite++;
      if (a.state === 'work' || a.state === 'meeting') d.working++;
      seen.set(a.dept, d);
    }
    return [...seen.values()].map(d => ({ ...d, utilization: d.total ? d.working / d.total : 0 }));
  }

  /** 当前时刻之后 horizonMin 分钟内，按 binMin 分桶的在楼人数趋势。
   *  通过「临时前推 — 统计 — 回到原位」实现，不改变对象当前状态。 */
  forecast(horizonMin = 120, binMin = 30) {
    const t0 = this.t;
    const bins = [];
    for (let m = 0; m <= horizonMin; m += binMin) {
      this._seekReadOnly(t0 + m);
      bins.push({
        offset: m,
        minute: t0 + m,
        onSite: this.agents.filter(a => a.state !== 'away').length,
        working: this.agents.filter(a => a.state === 'work').length,
        meeting: this.agents.filter(a => a.state === 'meeting').length,
        pantry: this.agents.filter(a => a.state === 'pantry' || a.state === 'lunch').length,
        lounge: this.agents.filter(a => a.state === 'lounge').length,
      });
    }
    this._seekReadOnly(t0);
    return bins;
  }

  /** 内部：把状态推进到 t 并把 this.t 一起还原（forecast 专用） */
  _seekReadOnly(t) {
    const target = Math.max(0, Math.min(24 * 60, t));
    if (target < this.t) {
      this.agents.forEach(a => this._resetAgent(a));
      this.t = 0;
    }
    const STEP = 0.25;                 // 预测只看大趋势，用大步长足够且更快
    while (this.t < target) {
      const dt = Math.min(STEP, target - this.t);
      this._step(dt);
      this.t += dt;
    }
  }
}
