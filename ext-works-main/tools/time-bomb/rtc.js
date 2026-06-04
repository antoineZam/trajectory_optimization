// WebRTC peer-to-peer sans serveur de signalisation.
// Topologie étoile : l'hôte maintient une RTCPeerConnection par joueur.
// L'échange SDP se fait par copier-coller (Base64 compressé).

const ICE_CONFIG = {
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
};

function waitForIceComplete(pc) {
  return new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') { resolve(); return; }
    pc.onicegatheringstatechange = () => {
      if (pc.iceGatheringState === 'complete') resolve();
    };
  });
}

// Compression DeflateRaw + Base64 — réduit le code de ~2000 à ~500 chars.
async function encode(sdp) {
  const json = JSON.stringify(sdp);
  const stream = new CompressionStream('deflate-raw');
  const writer = stream.writable.getWriter();
  writer.write(new TextEncoder().encode(json));
  writer.close();
  const buf = await new Response(stream.readable).arrayBuffer();
  return btoa(String.fromCharCode(...new Uint8Array(buf)));
}

async function decode(b64) {
  const clean = b64.replace(/\s+/g, '');
  const bytes = Uint8Array.from(atob(clean), c => c.charCodeAt(0));
  const stream = new DecompressionStream('deflate-raw');
  const writer = stream.writable.getWriter();
  writer.write(bytes);
  writer.close();
  const text = await new Response(stream.readable).text();
  return JSON.parse(text);
}

// ─── Hôte ────────────────────────────────────────────────────────────────────

export class TimeBombHost {
  constructor() {
    // Map<peerId, { pc: RTCPeerConnection, dc: RTCDataChannel }>
    this._peers = new Map();
    this._nextId = 1;

    // Callbacks à définir par le consommateur
    this.onMessage = null;       // (peerId, msg) => void
    this.onConnect = null;       // (peerId) => void
    this.onDisconnect = null;    // (peerId) => void
  }

  get playerIds() {
    return [...this._peers.keys()];
  }

  // Accepte une offre Base64 d'un joueur → retourne une réponse Base64.
  async acceptOffer(offerB64) {
    const peerId = `p${this._nextId++}`;
    const pc = new RTCPeerConnection(ICE_CONFIG);

    await pc.setRemoteDescription(await decode(offerB64));

    // Écoute le DataChannel créé par le joueur
    pc.ondatachannel = (evt) => {
      const dc = evt.channel;
      this._peers.set(peerId, { pc, dc });

      dc.onopen = () => {
        if (this.onConnect) this.onConnect(peerId);
      };

      dc.onclose = () => {
        this._peers.delete(peerId);
        if (this.onDisconnect) this.onDisconnect(peerId);
      };

      dc.onmessage = (e) => {
        if (this.onMessage) this.onMessage(peerId, JSON.parse(e.data));
      };
    };

    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    await waitForIceComplete(pc);

    return await encode(pc.localDescription);
  }

  send(peerId, msg) {
    const peer = this._peers.get(peerId);
    if (peer && peer.dc.readyState === 'open') {
      peer.dc.send(JSON.stringify(msg));
    }
  }

  broadcast(msg) {
    const raw = JSON.stringify(msg);
    for (const { dc } of this._peers.values()) {
      if (dc.readyState === 'open') dc.send(raw);
    }
  }

  isConnected(peerId) {
    const peer = this._peers.get(peerId);
    return peer ? peer.dc.readyState === 'open' : false;
  }
}

// ─── Joueur ──────────────────────────────────────────────────────────────────

export class TimeBombPeer {
  constructor() {
    this._pc = null;
    this._dc = null;

    this.onMessage = null;    // (msg) => void
    this.onConnect = null;    // () => void
    this.onDisconnect = null; // () => void
  }

  // Génère une offre Base64 à transmettre à l'hôte.
  async createOffer() {
    const pc = new RTCPeerConnection(ICE_CONFIG);
    this._pc = pc;

    const dc = pc.createDataChannel('game', { ordered: true });
    this._dc = dc;

    dc.onopen = () => {
      if (this.onConnect) this.onConnect();
    };
    dc.onclose = () => {
      if (this.onDisconnect) this.onDisconnect();
    };
    dc.onmessage = (e) => {
      if (this.onMessage) this.onMessage(JSON.parse(e.data));
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitForIceComplete(pc);

    return await encode(pc.localDescription);
  }

  // Reçoit la réponse Base64 de l'hôte → connexion établie.
  async acceptAnswer(answerB64) {
    await this._pc.setRemoteDescription(await decode(answerB64));
  }

  send(msg) {
    if (this._dc && this._dc.readyState === 'open') {
      this._dc.send(JSON.stringify(msg));
    }
  }

  get connected() {
    return this._dc ? this._dc.readyState === 'open' : false;
  }
}
