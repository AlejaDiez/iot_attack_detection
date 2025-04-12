import { FlowGenerator } from "@models/flow-generator";
import { Packet } from "@models/packet";

/**
 * Atacante Phantom.
 */
export class PhantomAttacker extends FlowGenerator {
    /** Ataques que puede realizar el atacante */
    public get attacks(): string[] {
        return Object.keys(this.library ?? {})
            .filter((e) => e.startsWith("atk_"))
            .map((e) => e.substring(4));
    }

    public override loadLibrary(library: any): void {
        const attacks = Object.keys(library).filter((e) =>
            e.startsWith("atk_"),
        );

        super.loadLibrary(library);
        for (const atk of attacks)
            this.library![atk] = (...args) =>
                library[atk](
                    {
                        mac: this.node.mac,
                        ip: this.node.ip,
                        name: this.node.name,
                        type: this.node.type as string,
                        send: (packet: Packet) => this.node.sendPacket(packet),
                    },
                    ...args,
                );
    }

    /**
     * Ejecuta un ataque de una biblioteca.
     *
     * @param fn Nombre de la función a ejecutar.
     * @param args Argumentos de la función.
     */
    public attack(fn: string, ...args: any[]): void {
        if (this.library && this.library[`atk_${fn}`])
            this.library[`atk_${fn}`](...args);
    }
}
