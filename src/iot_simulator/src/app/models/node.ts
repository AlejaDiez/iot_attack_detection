import { Position } from "@models/position";

/**
 * Enum que representa los diferentes tipos de nodos en el simulador IoT.
 */
export enum NodeType {
    ROUTER = "router",
    COMPUTER = "computer",
    IOT = "iot",
}

/**
 * Representa un nodo dentro del esquema de red.
 */
export abstract class Node {
    /* Identificador único del nodo */
    private _mac: string;
    public get mac(): string {
        return this._mac;
    }
    /* Tipo de nodo */
    private _type: NodeType;
    public get type(): NodeType {
        return this._type;
    }
    /* Indica si el nodo esta siendo arrastrado */
    public dragging: boolean;
    /* Indica si el nodo está seleccionado */
    public focused: boolean;
    /* Indica si el nodo está comunicando */
    public communicating: boolean;
    /* Indica si el nodo está en estado de inactividad */
    public get idle(): boolean {
        return this.focused === false && this.communicating === false;
    }
    /* Posición del nodo en el esquema de red */
    private _position: Position;
    public get position(): Position {
        return this._position;
    }

    /**
     * Crea una instancia de la clase Node.
     *
     * @param position Posición inicial del nodo.
     */
    public constructor(type: NodeType, position: Position) {
        this._mac = this._generateMacAddress();
        this._type = type;
        this.dragging = false;
        this.focused = false;
        this.communicating = false;
        this._position = position;
    }

    /**
     * Genera una dirección MAC aleatoria.
     *
     * @returns Dirección MAC aleatoria.
     */
    private _generateMacAddress(): string {
        return "XX:XX:XX:XX:XX:XX".replace(/X/g, () =>
            Math.floor(Math.random() * 16).toString(16)
        );
    }

    /**
     * Mueve el nodo en el esquema de red.
     *
     * @param x Desplazamiento horizontal.
     * @param y Desplazamiento vertical.
     */
    public move(x: number, y: number): void {
        this._position.x += x;
        this._position.y += y;
    }

    /**
     * Mueve el nodo a una posición concreta.
     *
     * @param x Coordenada x.
     * @param y Coordenada y.
     */
    public moveTo(x: number, y: number): void {
        this._position.x = x;
        this._position.y = y;
    }
}
