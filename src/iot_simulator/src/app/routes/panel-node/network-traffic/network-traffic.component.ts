import {
    Component,
    computed,
    inject,
    input,
    InputSignal,
    Signal,
} from "@angular/core";
import { toSignal } from "@angular/core/rxjs-interop";
import {
    FormControl,
    FormGroup,
    ReactiveFormsModule,
    Validators,
} from "@angular/forms";
import { HlmDialogService } from "@components/ui/ui-dialog-helm/src";
import { HlmLabelModule } from "@components/ui/ui-label-helm/src";
import { HlmMenuSeparatorComponent } from "@components/ui/ui-menu-helm/src";
import { Device } from "@models/device";
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
import { map, tap } from "rxjs";

@Component({
    imports: [
        ReactiveFormsModule,
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
        TranslateModule,
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
    protected readonly form: FormGroup = new FormGroup({
        command: new FormControl(null, [Validators.required]),
        target: new FormControl(null, [Validators.required]),
    });
    protected readonly multiple: Signal<boolean> = toSignal(
        this.form.get("command")!.valueChanges.pipe(
            tap(({ multiple }) => {
                const value = this.form.get("target")!.value;

                if (value) {
                    if (multiple && !Array.isArray(value))
                        this.form.get("target")!.setValue([value]);
                    else if (!multiple && Array.isArray(value))
                        this.form.get("target")!.setValue(value[0]);
                }
            }),
            map((e) => e.multiple),
        ),
        { initialValue: false },
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

    protected connect() {
        (this.node() as Device).connect(this.network.router!);
    }

    protected viewPacket(packet: Packet) {
        this.dialog.open(ShowPacketDialogComponent, { context: { packet } });
    }

    protected execute() {
        const { command, target } = this.form.value;

        switch (command.id) {
            case "ping":
                this.node().generator.ping(target);
                break;
            case "threeWayHandshake":
                this.node().generator.threeWayHandshake(target, 80, 80);
                break;
            default:
                this.node().generator.execute(
                    command.id,
                    ...(Array.isArray(target) ? target : [target]),
                );
                break;
        }
    }
}
