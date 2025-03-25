import { Node } from "@models/node";
import { Packet } from "@models/packet";

/**
 * Enumeración de los estados de transmisión de la conexión.
 */
export enum ConnectionStatus {
    IDLE,
    FORWARD,
    REVERSE,
}

/**
 * Representa una conexión entre un dispositivo y un router. A esta conexión se le puede asignar una latencia y permite comunicar ambos dispositivos.
 */
export class Connection {
    /** Cola de paquetes */
    private _queue: Packet[] = [];
    /** Indica si la conexión está procesando un paquete */
    private _processing: boolean = false;
    /** Router al que se conecta el dispositivo */
    public readonly node1: Node;
    /** Nodo para el que se establece la conexión */
    public readonly node2: Node;
    /** Latencia de la conexión */
    public readonly latency: number;
    private _status: ConnectionStatus;
    public get status(): ConnectionStatus {
        return this._status;
    }

    /**
     * Crea una instancia de la clase Connection.
     *
     * @param node1 Primer nodo de la conexión, suele ser un router.
     * @param node2 Segundo nodo de la conexión, suele ser un dispositivo.
     * @param latency Latencia de la conexión, se genera aleatoriamente si no se especifica.
     */
    public constructor(node1: Node, node2: Node, latency?: number) {
        this._queue = [];
        this._processing = false;
        this.node1 = node1;
        this.node2 = node2;
        this.latency =
            latency ?? Math.floor(Math.random() * (2000 - 500 + 1)) + 500;
        this._status = ConnectionStatus.IDLE;
    }

    /**
     * Procesa la cola de paquetes.
     */
    private async _processQueue(): Promise<void> {
        if (this._processing || this._queue.length === 0) return;

        this._processing = true;
        while (this._queue.length > 0) {
            const packet = this._queue.shift() as Packet;
            let reciver;

            if (packet.dstIP === this.node2.ip) {
                this._status = ConnectionStatus.FORWARD;
                reciver = this.node2;
            } else {
                this._status = ConnectionStatus.REVERSE;
                reciver = this.node1;
            }

            await new Promise((resolve) => setTimeout(resolve, this.latency));
            reciver.receivePacket(packet);
        }
        this._processing = false;
        this._status = ConnectionStatus.IDLE;
    }

    /**
     * Transmite un paquete desde el dispositivo al router o viceversa.
     *
     * @param packet Paquete a transmitir.
     */
    public spreadPacket(packet: Packet): void {
        this._queue.push(packet);
        this._processQueue();
    }

    /**
     * Convierte la conexión a un objeto plano.
     *
     * @returns Objeto plano con los datos de la conexión.
     */
    public toObject(): any {
        return this.latency;
    }
}
