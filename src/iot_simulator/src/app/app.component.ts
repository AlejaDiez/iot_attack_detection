import {
    AfterViewInit,
    Component,
    HostListener,
    inject,
    Signal,
    viewChild,
} from "@angular/core";
import { Router as NavigationRouter, RouterOutlet } from "@angular/router";
import { CanvasComponent } from "@components/canvas/canvas.component";
import { ConnectionComponent } from "@components/connection/connection.component";
import { MenuBarComponent } from "@components/menu-bar/menu-bar.component";
import { NodeComponent } from "@components/node/node.component";
import { HlmButtonModule } from "@components/ui/ui-button-helm/src";
import { Connection } from "@models/connection";
import { Node, NodeType } from "@models/node";
import { Router } from "@models/router";
import { NgIcon, provideIcons } from "@ng-icons/core";
import {
    lucideChevronsLeftRightEllipsis,
    lucideGhost,
    lucideSettings,
    lucideTrash,
} from "@ng-icons/lucide";
import { ConfigService } from "@services/config.service";
import { LibraryService } from "@services/library.service";
import { NetworkService } from "@services/network.service";
import { StateService } from "@services/state.service";
import { BrnContextMenuTriggerDirective } from "@spartan-ng/brain/menu";
import {
    HlmMenuComponent,
    HlmMenuGroupComponent,
    HlmMenuItemDirective,
    HlmMenuItemIconDirective,
    HlmMenuSeparatorComponent,
    HlmMenuShortcutComponent,
} from "@spartan-ng/ui-menu-helm";
import { HlmToasterComponent } from "@spartan-ng/ui-sonner-helm";
import { floatAnimation } from "./app.routes.transition";

@Component({ template: "" })
export class BlankComponent {}

@Component({
    selector: "app-root",
    imports: [
        RouterOutlet,
        BrnContextMenuTriggerDirective,
        CanvasComponent,
        ConnectionComponent,
        HlmButtonModule,
        HlmMenuComponent,
        HlmMenuGroupComponent,
        HlmMenuItemDirective,
        HlmMenuItemIconDirective,
        HlmMenuSeparatorComponent,
        HlmMenuShortcutComponent,
        HlmToasterComponent,
        MenuBarComponent,
        NodeComponent,
        NgIcon,
    ],
    providers: [
        provideIcons({
            lucideChevronsLeftRightEllipsis,
            lucideGhost,
            lucideSettings,
            lucideTrash,
        }),
    ],
    templateUrl: "app.component.html",
    animations: [floatAnimation],
})
export class AppComponent implements AfterViewInit {
    public readonly navigationRouter: NavigationRouter =
        inject(NavigationRouter);
    public readonly config: ConfigService = inject(ConfigService);
    public readonly state: StateService = inject(StateService);
    public readonly network: NetworkService = inject(NetworkService);
    public readonly library: LibraryService = inject(LibraryService);
    public readonly NodeType: typeof NodeType = NodeType;
    protected readonly canvas: Signal<CanvasComponent> =
        viewChild.required(CanvasComponent);
    protected get nodes(): Node[] {
        return this.network.nodes;
    }
    protected get router(): Router | undefined {
        return this.network.router;
    }
    protected get connections(): Connection[] {
        return this.router?.connections ?? [];
    }

    public ngAfterViewInit(): void {
        this.onCenter("instant");
    }

    protected getAnimationData(outlet: RouterOutlet) {
        if (outlet && !outlet.isActivated) return "_";
        return outlet.activatedRoute.snapshot.params["mac"] ?? "_";
    }

    @HostListener("document:keydown.meta.n", ["$event"])
    protected onNewFile(event?: Event) {
        event?.preventDefault();
        this.network.new();
    }

    @HostListener("document:keydown.meta.o", ["$event"])
    protected onOpenFile(event?: Event) {
        event?.preventDefault();
        this.network.loadFromFile();
    }

    protected onLoadExternalLibrary(event?: Event) {
        event?.preventDefault();
        this.library.loadFromFile();
    }

    @HostListener("document:keydown.meta.shift.s", ["$event"])
    protected onSaveFile(event?: Event) {
        event?.preventDefault();
        this.network.saveToFile();
    }

    @HostListener("document:keydown.meta.z", ["$event"])
    protected onUndo(event?: Event) {
        event?.preventDefault();
        this.state.undo();
    }

    @HostListener("document:keydown.meta.shift.z", ["$event"])
    protected onRedo(event?: Event) {
        event?.preventDefault();
        this.state.redo();
    }

    @HostListener("document:keydown.meta.shift.r", [
        "NodeType.RouterTypes",
        "$event",
    ])
    @HostListener("document:keydown.meta.shift.d", [
        "NodeType.DeviceTypes",
        "$event",
    ])
    protected onInsertNode(
        type?: NodeType | readonly NodeType[] | NodeType[],
        event?: Event,
    ) {
        const { scrollWidth, scrollHeight, scrollLeft, scrollTop } =
            this.canvas().frame().nativeElement;
        const { left, top, width, height } = this.canvas()
            .frame()
            .nativeElement.getBoundingClientRect();
        const centerX = scrollWidth / 2;
        const centerY = scrollHeight / 2;

        event?.preventDefault();
        if (event instanceof MouseEvent) {
            const { clientX, clientY } = event;

            this.network.addNode(type, {
                x: clientX - left + scrollLeft - centerX,
                y: centerY - (clientY - top + scrollTop),
            });
        } else {
            this.network.addNode(type, {
                x: width / 2 - left + scrollLeft - centerX,
                y: centerY - (height / 2 - top + scrollTop),
            });
        }
    }

    @HostListener("document:keydown.alt.0", ["'smooth'", "$event"])
    protected onCenter(behavior: ScrollBehavior, event?: Event) {
        const frame = this.canvas().frame().nativeElement;
        const { scrollHeight, clientHeight, scrollWidth, clientWidth } = frame;

        event?.preventDefault();
        frame.scrollTo({
            top: (scrollHeight - clientHeight) / 2,
            left: (scrollWidth - clientWidth) / 2,
            behavior,
        });
    }

    protected onDeleteNode(mac: string, event?: Event) {
        event?.preventDefault();
        this.network.deleteNode(mac);
    }

    protected onNodeTraffic(mac: string, event?: Event) {
        event?.preventDefault();
        this.navigationRouter.navigate([mac, "network-traffic"]);
    }

    protected onNodeAttack(mac: string, event?: Event) {
        event?.preventDefault();
        this.navigationRouter.navigate([mac, "attack"]);
    }

    protected onNodeConfig(mac: string, event?: Event) {
        event?.preventDefault();
        this.navigationRouter.navigate([mac, "configuration"]);
    }
}
