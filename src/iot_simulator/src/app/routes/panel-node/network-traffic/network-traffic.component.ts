import { CommonModule } from "@angular/common";
import {
    Component,
    computed,
    effect,
    inject,
    input,
    InputSignal,
    model,
    ModelSignal,
    Signal,
} from "@angular/core";
import { FormsModule } from "@angular/forms";
import { HlmDialogService } from "@components/ui/ui-dialog-helm/src";
import { HlmLabelModule } from "@components/ui/ui-label-helm/src";
import { HlmMenuSeparatorComponent } from "@components/ui/ui-menu-helm/src";
import { Device } from "@models/device";
import { Command } from "@models/flow-generator";
import { Node, NodeType } from "@models/node";
import { Packet } from "@models/packet";
import { NgIcon, provideIcons } from "@ng-icons/core";
import {
    lucideForward,
    lucideHourglass,
    lucideRepeat,
    lucideReply,
    lucideTrafficCone,
    lucideUnplug,
} from "@ng-icons/lucide";
import { TranslateModule } from "@ngx-translate/core";
import { ShowPacketDialogComponent } from "@routes/dialogs/show-packet-dialog.component";
import { NetworkService } from "@services/network.service";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { BrnTableModule } from "@spartan-ng/brain/table";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmInputModule } from "@spartan-ng/ui-input-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { HlmTableModule } from "@spartan-ng/ui-table-helm";
import { HlmTabsModule } from "@spartan-ng/ui-tabs-helm";

@Component({
    imports: [
        CommonModule,
        FormsModule,
        TranslateModule,
        BrnSelectModule,
        BrnTableModule,
        HlmButtonModule,
        HlmInputModule,
        HlmLabelModule,
        HlmMenuSeparatorComponent,
        HlmSelectModule,
        HlmTableModule,
        HlmTabsModule,
        NgIcon,
    ],
    providers: [
        provideIcons({
            lucideForward,
            lucideHourglass,
            lucideUnplug,
            lucideRepeat,
            lucideReply,
            lucideTrafficCone,
        }),
    ],
    templateUrl: "network-traffic.component.html",
    host: { class: "flex flex-col gap-4" },
})
export class NetworkTrafficComponent {
    public readonly dialog: HlmDialogService = inject(HlmDialogService);
    public readonly network: NetworkService = inject(NetworkService);
    protected readonly node: InputSignal<Node> = input.required<Node>();
    public readonly NodeType: typeof NodeType = NodeType;
    protected readonly displayedColumns: Signal<string[]> = computed(() => [
        "icon",
        "source",
        "destination",
        "size",
        "data",
    ]);
    protected readonly command: ModelSignal<Command | null> =
        model<Command | null>(null);
    protected readonly target: ModelSignal<string | string[] | null> = model<
        string | string[] | null
    >(null);
    protected readonly multipleTargets: Signal<boolean> = computed(
        () => this.command()?.multiple ?? false,
    );
    protected get canConnect(): boolean {
        return this.network.router !== undefined;
    }
    protected get commands(): any[] {
        const internalCommands = this.node().generator.internalCommands;
        const externalCommands = this.node().generator.externalCommands;

        if (internalCommands.length > 0 && externalCommands.length > 0)
            return [...internalCommands, "---", ...externalCommands];
        return [...internalCommands, ...externalCommands];
    }
    protected get connectedNodes(): Node[] {
        return this.network.getConnectedNodes(this.node().mac);
    }

    public constructor() {
        effect(() => {
            if (this.multipleTargets())
                this.target.update((value) => {
                    if (value && !Array.isArray(value))
                        return [value as string];
                    return value;
                });
            else
                this.target.update((value) => {
                    if (value && Array.isArray(value)) return value[0];
                    return value;
                });
        });
    }

    protected connect() {
        (this.node() as Device).connect(this.network.router!);
    }

    protected viewPacket(packet: Packet) {
        this.dialog.open(ShowPacketDialogComponent, { context: { packet } });
    }

    protected execute() {
        switch (this.command()!.id) {
            case "ping":
                this.node().generator.ping(this.target() as string);
                break;
            case "threeWayHandshake":
                this.node().generator.threeWayHandshake(
                    this.target() as string,
                    80,
                    80,
                );
                break;
            default:
                this.node().generator.execute(
                    this.command()!.id,
                    ...(Array.isArray(this.target()!)
                        ? this.target()!
                        : [this.target()!]),
                );
                break;
        }
    }
}
